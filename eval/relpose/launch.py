"""Camera pose for LinearVGGT and StreamVGGT (CUT3R / TTT3R protocol, ported for VGGT-family models).

Per sequence: frames resized to long side 518 (multiple of 14, no crop, as the video-depth eval), streamed
through model.inference under bf16 autocast (as every other eval here), camera poses from the pose
encoding (camera-from-world, frame 0 = world) inverted to camera-to-world, then evo ATE (Sim(3) Umeyama,
translation RMSE) and RPE trans / rot, with eval.relpose.evo_utils.eval_metrics from TTT3R (edd6d8c).
Frames whose GT pose is not finite (6 ScanNet test scenes have -inf poses) are dropped from both
trajectories before the metrics; the count is logged. A CUDA OOM marks the sequence as OOM and continues.
Writes <output_dir>/<seq>/pred_traj.txt, <seq>_eval_metric.txt, <seq>.png, results.jsonl, summary.json.

  python eval/relpose/launch.py --model_name LinearVGGT --weights CKPT --eval_dataset tum \
      --output_dir OUT
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

from eval.model_loader import add_model_args, load_eval_model, prefetch_files
from eval.relpose.metadata import dataset_metadata


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_dataset", required=True, choices=sorted(dataset_metadata))
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="diagnostic: keep every K-th frame of each sequence (and its GT pose)")
    parser.add_argument("--max_frames", type=int, default=0, help="diagnostic: then keep the first N frames")
    parser.add_argument("--resize_hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="diagnostic: stretch the frames to H x W (multiples of 14) before inference")
    parser.add_argument("--max_seqs", type=int, default=None, help="only the first N sequences (smoke tests)")
    return add_model_args(parser)


def build_views(paths, size, resize_hw=None):
    from dust3r_utils.image import load_images_for_eval
    prefetch_files(paths)
    images = load_images_for_eval(paths, size=size, crop=False, verbose=False)
    views = [dict(img=((im["img"] + 1.0) / 2.0).cuda(), true_shape=torch.from_numpy(im["true_shape"]),
                  idx=i, instance=str(i)) for i, im in enumerate(images)]
    if resize_hw:  # diagnostic: stretch every frame to H x W (e.g. the training token count), aspect not kept
        for view in views:
            view["img"] = torch.nn.functional.interpolate(view["img"], size=tuple(resize_hw), mode="bilinear",
                                                          antialias=True, align_corners=False)
            view["true_shape"] = torch.tensor([list(resize_hw)])
    return views


@torch.no_grad()
def predict_c2w(model, views):
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ress = model.inference(views).ress
    pose_enc = torch.stack([r["camera_pose"] for r in ress], dim=1).float()       # [1, T, 9]
    height, width = views[0]["img"].shape[-2:]
    extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(height, width))
    w2c = torch.eye(4, device=extrinsic.device).repeat(extrinsic.shape[1], 1, 1)
    w2c[:, :3] = extrinsic[0]
    return torch.linalg.inv(w2c).cpu().numpy()


def load_replica_finite(path):
    """ScanNet pose_N.txt (row-major 4x4 camera-to-world per line) -> (N, 7) xyz + quat wxyz, timestamps.
    Rows with a non-finite pose become NaN (evo's own loader crashes on them while computing quaternions)."""
    from scipy.spatial.transform import Rotation
    poses = np.loadtxt(path).reshape(-1, 4, 4)
    out = np.full((len(poses), 7), np.nan)
    finite = np.isfinite(poses).all((1, 2))
    if finite.any():
        quat = Rotation.from_matrix(poses[finite, :3, :3]).as_quat()           # x y z w
        out[finite] = np.column_stack((poses[finite, :3, 3], quat[:, 3], quat[:, :3]))
    return out, np.arange(len(poses)).astype(float)


def main(args):
    from eval.relpose.evo_utils import eval_metrics, load_traj, plot_trajectory
    from eval.relpose.utils import get_tum_poses, save_tum_poses

    meta = dataset_metadata[args.eval_dataset]
    img_path = meta["img_path"]
    seqs = meta["seq_list"] or sorted(s for s in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, s)))
    seqs = seqs[:args.max_seqs] if args.max_seqs else seqs
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_eval_model(args, "cuda")
    rows = []
    results_path = os.path.join(args.output_dir, "results.jsonl")
    open(results_path, "w").close()
    for seq in seqs:
        dir_path = meta["dir_path_func"](img_path, seq)
        paths = sorted(os.path.join(dir_path, n) for n in os.listdir(dir_path))
        keep = list(range(0, len(paths), args.frame_stride))[:args.max_frames or None]
        paths = [paths[i] for i in keep]
        row = dict(seq=seq, frames=len(paths))
        try:
            views = build_views(paths, args.size, args.resize_hw)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
            c2w = predict_c2w(model, views)
            torch.cuda.synchronize()
            row.update(seconds=time.time() - start, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            row["fps"] = len(paths) / row["seconds"]
            del views
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row["oom"] = True
            print(f"{args.eval_dataset} {seq}: OOM at {len(paths)} frames", flush=True)
            rows.append(row)
            with open(results_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            continue
        os.makedirs(os.path.join(args.output_dir, seq), exist_ok=True)
        save_tum_poses(c2w, os.path.join(args.output_dir, seq, "pred_traj.txt"))
        gt_file = meta["gt_traj_func"](img_path, meta.get("anno_path"), seq)
        if meta["traj_format"] == "replica":
            gt_poses, gt_times = load_replica_finite(gt_file)
        else:
            gt_poses, gt_times = load_traj(gt_file, traj_format=meta["traj_format"])
        if args.frame_stride != 1 or args.max_frames:
            gt_poses, gt_times = gt_poses[keep], gt_times[keep]
        if len(gt_poses) != len(c2w):
            raise ValueError(f"{seq}: {len(c2w)} frames but {len(gt_poses)} GT poses")
        finite = np.isfinite(gt_poses).all(1)
        row["invalid_gt"] = int((~finite).sum())
        pred_poses, _ = get_tum_poses(c2w)
        pred = [pred_poses[finite], gt_times[finite]]
        gt = [gt_poses[finite], gt_times[finite]]
        try:
            ate, rpe_trans, rpe_rot = eval_metrics(pred, gt, seq=seq,
                                                   filename=os.path.join(args.output_dir, f"{seq}_eval_metric.txt"))
            plot_trajectory(pred, gt, title=seq, filename=os.path.join(args.output_dir, f"{seq}.png"))
            row.update(ate=float(ate), rpe_trans=float(rpe_trans), rpe_rot=float(rpe_rot))
        except Exception as error:  # evo raises on degenerate trajectories
            row["metric_error"] = str(error)
        print(f"{args.eval_dataset} {seq}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                         for k, v in row.items() if k != "seq"), flush=True)
        rows.append(row)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    scored = [r for r in rows if "ate" in r]
    summary = dict(dataset=args.eval_dataset, model=args.model_name, weights=args.weights,
                   sequences=len(rows), scored=len(scored), oom=sum("oom" in r for r in rows),
                   **{k: float(np.mean([r[k] for r in scored])) if scored else None
                      for k in ("ate", "rpe_trans", "rpe_rot", "fps", "peak_gib")})
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
