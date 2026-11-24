#!/usr/bin/env bash
# Video depth on Sintel, Bonn and KITTI (AbsRel, delta < 1.25), scale and scale & shift alignment.
#   bash scripts/eval_video_depth.sh ckpt/linearvggt.pt [LinearVGGT|StreamVGGT] [extra launch.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
weights="${1:?checkpoint}"; model="${2:-LinearVGGT}"; shift; shift || true
for data in sintel bonn kitti; do
  out="eval_results/video_depth/${model}/${data}"
  python eval/video_depth/launch.py --weights "${weights}" --model_name "${model}" --output_dir "${out}" \
    --eval_dataset "${data}" --size 518 "$@"
  for align in scale "scale&shift"; do
    python eval/video_depth/eval_depth.py --output_dir "${out}" --eval_dataset "${data}" --align "${align}"
  done
done
