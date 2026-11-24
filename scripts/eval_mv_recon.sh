#!/usr/bin/env bash
# Multi-view reconstruction on 7-Scenes and NRGBD (Acc, Comp, NC), sparse and dense views.
#   bash scripts/eval_mv_recon.sh ckpt/linearvggt.pt [LinearVGGT|StreamVGGT] [extra launch.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
weights="${1:?checkpoint}"; model="${2:-LinearVGGT}"; shift; shift || true
for protocol in "sparse 200 500" "dense 20 50"; do
  read -r name kf7 kfn <<< "${protocol}"
  out="eval_results/mv_recon/${model}/${name}"
  python eval/mv_recon/launch.py --weights "${weights}" --model_name "${model}" --output_dir "${out}" \
    --kf_every_7scenes "${kf7}" --kf_every_nrgbd "${kfn}" "$@"
  python eval/mv_recon/summarize.py "${out}"
done
