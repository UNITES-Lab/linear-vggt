"""Streaming speed/memory vs sequence length: teacher, eager student, FastStreamer student.

Protocol: one GPU, the same random 518x392 frames, batch 1, a
synchronize after every frame, per-frame CUDA-event time of the whole frame (aggregator + all heads), memory after
each frame. Rows at frames 0, 7, 31, 63, 127, 255, 511, 1023 are printed; every frame goes to the JSONL output.
  teacher      official StreamVGGT streaming path (aggregator + camera KV caches), until OOM
  eager        LinearVGGT, plain PyTorch (inference.streaming.predict_images per frame), token eviction as in
               the checkpoint; the two FastStreamer rows run without token eviction (not implemented there)
  fast-exact   FastStreamer(exact=True, fused=False, graphs=True): bitwise identical to eager without eviction
  fast         FastStreamer(exact=True, fused=True, graphs=True): the eval --fast_stream setting
Memory: state = allocated now minus allocated before the stream (caches/recurrent state); peak = max allocated
minus that base; reserved = max reserved by the allocator (whole process, includes weights and the CUDA-graph
pools, which `allocated` does not see once the capture is done).
  python scripts/benchmark_speed.py --teacher ckpt/streamvggt.pth --student ckpt/linearvggt.pt --frames 1024 \
      --output outputs/speed.jsonl [--skip-teacher]
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import linearvggt  # noqa: E402,F401  (makes third_party/ importable)
import torch

from linearvggt.inference.fast_stream import FastStreamer
from linearvggt.models import linearvggt as hybrid_student
from streamvggt.models.streamvggt import StreamVGGT

REPORT = {0, 7, 31, 63, 127, 255, 511, 1023, 2047}


@torch.no_grad()
def teacher_steps(model, frames):
    cache = [None] * model.aggregator.depth
    camera_cache = [None] * model.camera_head.trunk_depth
    for i, images in enumerate(frames):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, start, cache = model.aggregator(images, past_key_values=cache, use_cache=True, past_frame_idx=i)
        with torch.autocast("cuda", enabled=False):
            tokens = [x.float() for x in tokens]
            _, camera_cache = model.camera_head(tokens, past_key_values_camera=camera_cache, use_cache=True)
            model.depth_head(tokens, images=images, patch_start_idx=start)
            model.point_head(tokens, images=images, patch_start_idx=start)
        yield


@torch.no_grad()
def eager_steps(model, frames):
    cache, camera_cache = None, None
    for i, images in enumerate(frames):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, start, cache = model.aggregator(images, past_key_values=cache, use_cache=True, past_frame_idx=i)
        with torch.autocast("cuda", enabled=False):
            tokens = [x.float() for x in tokens]
            model.depth_head(tokens, images=images, patch_start_idx=start)
            model.point_head(tokens, images=images, patch_start_idx=start)
            _, camera_cache = model.camera_head.stream_step(tokens[-1], camera_cache)
        yield


def fast_steps(streamer):
    def steps(_, frames):
        streamer.reset()
        for images in frames:
            streamer.step(images)
            yield
    return steps


def measure(name, model, steps, frames, warmup=4):
    for _ in steps(model, frames[:warmup]):
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    rows = []
    try:
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        for t, _ in enumerate(steps(model, frames)):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            rows.append(dict(model=name, frame=t, total_ms=start.elapsed_time(end),
                             state_gib=(torch.cuda.memory_allocated() - base) / 2**30,
                             peak_gib=(torch.cuda.max_memory_allocated() - base) / 2**30,
                             reserved_gib=torch.cuda.max_memory_reserved() / 2**30, weights_gib=base / 2**30))
            start = end
            if t in REPORT or t == len(frames) - 1:
                r = rows[-1]
                print(f"{name:11s} frame {t:5d}: {r['total_ms']:8.2f} ms  state {r['state_gib']:6.2f} GiB  "
                      f"peak {r['peak_gib']:6.2f} GiB  reserved {r['reserved_gib']:6.2f} GiB", flush=True)
    except torch.cuda.OutOfMemoryError:
        rows.append(dict(model=name, frame=len(rows), oom=True))
        print(f"{name:11s} OOM at frame {len(rows) - 1}", flush=True)
    torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="ckpt/streamvggt.pth", help="StreamVGGT checkpoint")
    p.add_argument("--student", default="ckpt/linearvggt.pt", help="LinearVGGT checkpoint")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    frames = [torch.rand(1, 1, 3, 392, 518, device="cuda") for _ in range(args.frames)]
    print(torch.cuda.get_device_name(), torch.__version__, f"{args.frames} frames", flush=True)
    rows = []
    if not args.skip_teacher:
        teacher = StreamVGGT()
        teacher.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True, mmap=True), strict=True)
        teacher = teacher.eval().cuda()
        rows += measure("teacher", teacher, teacher_steps, frames)
        del teacher
        torch.cuda.empty_cache()
    student = hybrid_student.load_student(args.student, "cuda")
    rows += measure("eager", student, eager_steps, frames)      # the checkpoint's own setting (token eviction on)
    for layer in student.hybrid_config["sparse_layers"]:        # FastStreamer does not implement token eviction
        student.aggregator.global_blocks[layer].attn.evict_budget = None
    for name, fused in (("fast-exact", False), ("fast", True)):
        streamer = FastStreamer(student, exact=True, fused=fused, graphs=True)
        rows += measure(name, student, fast_steps(streamer), frames)
        del streamer
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for name in dict.fromkeys(r["model"] for r in rows):
        ok = [r for r in rows if r["model"] == name and not r.get("oom")]
        mid = [r["total_ms"] for r in ok if 16 <= r["frame"]]
        if mid:
            print(f"{name:11s} mean ms/frame over frames 16..{ok[-1]['frame']}: {sum(mid) / len(mid):.1f}", flush=True)


if __name__ == "__main__":
    main()
