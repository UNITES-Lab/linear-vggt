#!/usr/bin/env bash
# Single-frame depth on Sintel, Bonn, KITTI and NYU-v2.
#   bash scripts/eval_monodepth.sh ckpt/linearvggt.pt [LinearVGGT|StreamVGGT] [extra launch.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
weights="${1:?checkpoint}"; model="${2:-LinearVGGT}"; shift; shift || true
for data in sintel bonn kitti nyu; do
  out="eval_results/monodepth/${model}/${data}"
  python eval/monodepth/launch.py --weights "${weights}" --model_name "${model}" --output_dir "${out}" \
    --eval_dataset "${data}" "$@"
  python eval/monodepth/eval_metrics.py --output_dir "${out}" --eval_dataset "${data}"
done
