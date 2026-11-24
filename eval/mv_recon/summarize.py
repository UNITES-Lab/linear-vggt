"""Verify full benchmark coverage and produce machine-readable metrics."""
import argparse
import json
import math
from pathlib import Path
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--datasets", default="7scenes,NRGBD",
                    help="comma list of the datasets that were run (mv_recon --datasets); each must be complete")
args = parser.parse_args()
COUNTS = {"7scenes": 18, "NRGBD": 9}
summary = {}
for dataset in args.datasets.split(","):
    count = COUNTS[dataset]
    rows = []
    logs = sorted(args.output_dir.joinpath(dataset).glob("logs_[0-9]*.txt"))   # one per mv_recon process
    for line in "\n".join(f.read_text() for f in logs).splitlines():
        scene = re.match(r"Idx:\s*([^,]+),", line)
        if not scene:
            continue
        values = dict(re.findall(r"(Acc|Comp|NC1|NC2|Acc_med|Compc_med|NC1c_med|NC2c_med):\s*([^,\s]+)", line))
        values = {k: float(v) for k, v in values.items()}
        if len(values) != 8 or not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Incomplete or nonfinite metrics: {line}")
        rows.append(dict(scene=scene.group(1), **values))
    if len(rows) != count or len({r["scene"] for r in rows}) != count:
        raise ValueError(f"{dataset}: expected {count} unique results, got {len(rows)}")
    means = {k: sum(r[k] for r in rows) / count for k in rows[0] if k != "scene"}
    means["NC"] = (means["NC1"] + means["NC2"]) / 2
    summary[dataset] = dict(count=count, means=means, sequences=rows)
(args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: {"count": v["count"], **v["means"]} for k, v in summary.items()}, indent=2))
