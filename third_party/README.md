# Third-party code

This directory contains code from other projects. LinearVGGT imports it as normal Python packages
(`pip install -e .` installs them; a checkout without installation finds them through `import linearvggt`).

| Directory | Package | Source | License |
|---|---|---|---|
| `streamvggt/` | `streamvggt` | [StreamVGGT](https://github.com/wzzheng/StreamVGGT), which is built on [VGGT](https://github.com/facebookresearch/vggt) and DINOv2 | CC BY-NC-SA 4.0 (StreamVGGT); see the header of each file |
| `dust3r_utils/` | `dust3r_utils` | [DUSt3R](https://github.com/naver/dust3r) / [CUT3R](https://github.com/CUT3R/CUT3R): image loading, geometry and camera utilities for the evaluation | CC BY-NC-SA 4.0 |

## Changes to `streamvggt/`

The model code is the code of StreamVGGT with these changes:

- `models/aggregator.py`: the streaming path accepts caches that are not key/value tensors (the state of a Mamba2 mixer, the cache of a sparse attention layer) and reads the number of streamed frames from them. It also supports a batch size larger than 1 and raises an error for a wrong input in place of a print.
- `heads/camera_head.py`: `stream_step` keeps one cache for each refinement iteration, so that streaming gives the same result as the batch path.
- `layers/rope.py`: the frequency table has a fixed length. This removes one device-to-host synchronization in each attention call and does not change the values.

LinearVGGT loads the weights of the encoder, the frame attention layers and the heads without a change of the parameter names, so a StreamVGGT checkpoint loads into `streamvggt.models.streamvggt.StreamVGGT`.
