"""Batch and streaming inference with identical outputs (shared by the model and the training code)."""
import torch


def predict_images(model, images, streaming=False, output_buffers=None):
    """Images: [B,T,3,H,W], RGB float in [0,1]; inference caller uses no_grad.

    Streaming caches the aggregator and the camera head. The camera cache is kept
    per refinement iteration (CameraHead.stream_step), which preserves the batch
    head's semantics (the legacy camera KV path mixes refinement iterations).

    output_buffers (streaming only): host tensors to stream the outputs into, as from host_output_buffers().
    Each frame's outputs are copied with non_blocking copies as soon as they exist (no per-frame
    synchronisation), so GPU memory holds only the model state; the buffers are returned after one sync.
    """
    if images.ndim != 5 or images.shape[1] < 1:
        raise ValueError("Expected nonempty [B,T,3,H,W] images")
    if output_buffers is not None and not streaming:
        raise ValueError("output_buffers needs streaming=True")
    streamer = getattr(model, "fast_streamer", None)       # LinearVGGT.enable_fast_stream()
    if streaming and streamer is not None:
        return streamer.predict(images, output_buffers)
    cache, camera_cache = None, None
    outputs = []
    chunks = images.split(1, dim=1) if streaming else (images,)
    for t, chunk in enumerate(chunks):
        if streaming:
            features, start, cache = model.aggregator(
                chunk, past_key_values=cache, use_cache=True, past_frame_idx=t)
        else:
            features, start = model.aggregator(chunk)
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = [x.float() for x in features]
            depth, depth_conf = model.depth_head(features, images=chunk, patch_start_idx=start)
            points, point_conf = model.point_head(features, images=chunk, patch_start_idx=start)
            if streaming:
                pose, camera_cache = model.camera_head.stream_step(features[-1], camera_cache)
            else:
                pose = model.camera_head(features)[-1]
        frame = dict(depth=depth, points=points, pose=pose, depth_conf=depth_conf, point_conf=point_conf)
        if output_buffers is None:
            outputs.append(frame)
        else:
            for key, value in frame.items():
                output_buffers[key][:, t].copy_(value[:, 0], non_blocking=True)
    if output_buffers is not None:
        torch.cuda.synchronize(images.device)
        return output_buffers
    return {key: torch.cat([o[key] for o in outputs], dim=1) for key in outputs[0]}


def host_output_buffers(images):
    """Pinned host buffers for predict_images(..., output_buffers=...) on [B,T,3,H,W] images (fp32 heads).
    Allocating pinned memory is slow (seconds for long streams); allocate once, outside timed code."""
    b, t, _, h, w = images.shape
    shapes = dict(depth=(h, w, 1), points=(h, w, 3), pose=(9,), depth_conf=(h, w), point_conf=(h, w))
    return {key: torch.empty((b, t) + shape, dtype=torch.float32, pin_memory=True) for key, shape in shapes.items()}
