#!/usr/bin/env bash
# Camera pose on Sintel, TUM-dynamics and ScanNet (ATE, RPE translation, RPE rotation).
#   bash scripts/eval_pose.sh ckpt/linearvggt.pt [LinearVGGT|StreamVGGT] [extra launch.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
weights="${1:?checkpoint}"; model="${2:-LinearVGGT}"; shift; shift || true
for data in sintel tum scannet; do
  python eval/relpose/launch.py --weights "${weights}" --model_name "${model}" \
    --output_dir "eval_results/relpose/${model}/${data}" --eval_dataset "${data}" "$@"
done
