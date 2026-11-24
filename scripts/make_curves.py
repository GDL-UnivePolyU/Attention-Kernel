import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

COLUMNS = [("lin", "feat_linear"), ("rbf", "feat_rbf"), ("mix", "feat_linear+rbf"),
           ("one", "K1"), ("onefeat", "K1+feat"), ("sum", "K1+K2"), ("sumfeat", "K1+K2+feat")]
DATASETS = ("awa2", "celeba", "tox21")
XALL = 8192


def curve(rows: list[dict], metric: str) -> list[tuple]:
    """(k, score per column) for every shot setting of the rows."""
    cell = defaultdict(list)   # (method, shots, seed, target) -> scores of the repeats
    for r in rows:
        if r.get(metric) is not None:
            cell[(r["method"], r["shots"], r["seed"], r["task"])].append(r[metric])
    shots = sorted({s for (_, s, _, _) in cell}, key=lambda s: XALL if s == "all" else int(s))
    out = []
    for s in shots:
        line = []
        for _, m in COLUMNS:
            per_seed = defaultdict(list)
            for (mm, ss, sd, _), v in cell.items():
                if mm == m and ss == s:
                    per_seed[sd].append(np.nanmean(v))
            line.append(np.mean([np.mean(x) for x in per_seed.values()]) if per_seed else np.nan)
        out.append((XALL if s == "all" else int(s), line))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default=os.path.join("results", "figures"))
    for ds in DATASETS:
        p.add_argument(f"--{ds}_tag", default="", help=f"only the files {ds}_<mode>_<tag>*.jsonl (e.g. n8000_)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    methods = {m for _, m in COLUMNS}
    for ds in DATASETS:
        tag = getattr(a, f"{ds}_tag")
        files = [f for mode in ("attention", "features")
                 for f in glob.glob(os.path.join(a.results, f"{ds}_{mode}{'_' + tag if tag else ''}*.jsonl"))]
        rows = [r for f in files for r in map(json.loads, open(f))
                if r.get("shots") is not None and r["method"] in methods]
        if not rows:
            print(ds, "no results")
            continue
        path = os.path.join(a.out, f"curve_{ds}.dat")
        with open(path, "w") as f:
            f.write("k " + " ".join(c for c, _ in COLUMNS) + "\n")
            for k, line in curve(rows, "auc" if ds == "tox21" else "bacc"):
                f.write(f"{k} " + " ".join("nan" if np.isnan(v) else f"{v:.4f}" for v in line) + "\n")
        print(f"{ds}: {len(files)} files -> {path}")


if __name__ == "__main__":
    main()
