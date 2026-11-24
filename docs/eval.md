# Evaluation

The evaluation code follows [StreamVGGT](https://github.com/wzzheng/StreamVGGT), [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/docs/eval.md), [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [TTT3R](https://github.com/Inception3D/TTT3R). Run all commands from the repository root.

## Datasets

Prepare Sintel, Bonn, KITTI, NYU-v2, TUM-dynamics and ScanNet as [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) tells you, and 7-Scenes and Neural-RGBD as [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) tells you.

```
data/eval
├── 7scenes
├── neural_rgbd
├── sintel
├── bonn
├── kitti
├── nyu_v2
├── tum
└── scannetv2
```

## Multi-view reconstruction

```bash
bash scripts/eval_mv_recon.sh CKPT
```

The script runs two protocols and writes `eval_results/mv_recon/<model>/<protocol>/summary.json`:

| Protocol | Keyframe interval, 7-Scenes | Keyframe interval, NRGBD |
|---|---|---|
| `sparse` | 200 | 500 |
| `dense` | 20 | 50 |

For other protocols, call the launcher directly:

```bash
python eval/mv_recon/launch.py --weights CKPT --model_name LinearVGGT --output_dir OUT \
    --datasets 7scenes --kf_every_7scenes 2 --max_frames 300      # long streams
python eval/mv_recon/summarize.py OUT --datasets 7scenes
```

## Video depth, monocular depth and camera pose

```bash
bash scripts/eval_video_depth.sh CKPT
bash scripts/eval_monodepth.sh   CKPT
bash scripts/eval_pose.sh        CKPT
```

Results are in `eval_results/<task>/<model>/<dataset>/`.

## Model options

All launchers accept these options:

| Option | Function |
|---|---|
| `--model_name LinearVGGT\|StreamVGGT` | The model. `StreamVGGT` evaluates the teacher with the same pipeline |
| `--fast_stream` | Use the efficient streaming inference (CUDA graphs and fused kernels). It sets the token indexer to off, because this mode does not implement it |

## Speed and memory

```bash
python scripts/benchmark_speed.py --teacher STREAMVGGT_CKPT --student CKPT \
    --frames 1024 --output outputs/speed.jsonl
```

The script streams random 518×392 frames with batch size 1. It records the latency of each frame (CUDA events, heads included) and the GPU memory for the teacher, for LinearVGGT in plain PyTorch, and for LinearVGGT with the efficient streaming inference. Add `--skip-teacher` to measure only LinearVGGT.
