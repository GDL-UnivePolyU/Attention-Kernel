from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from functools import partial

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from attnkernel import data as D
from attnkernel import kernels as KK
from attnkernel.train import TrainConfig, random_split, svm_eval, train_model

SHOTS = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "all"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["awa2", "celeba", "tox21"])
    p.add_argument("--mode", required=True, choices=["attention", "features"])
    p.add_argument("--n_samples", type=int, default=None, help="random subset per seed (default: all samples)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--gamma", type=float, default=KK.GAMMA, help="bandwidth factor of the attention kernel")
    p.add_argument("--shots", nargs="+", default=SHOTS)
    p.add_argument("--repeats", type=int, default=5, help="random orders for k <= 64 (3 for k <= 256, 1 above)")
    p.add_argument("--targets", nargs="+", default=None, help="only these target tasks")
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="results")
    p.add_argument("--verbose", action="store_true")
    for f, v in asdict(TrainConfig()).items():
        p.add_argument(f"--{f}", type=type(v), default=v)
    a = p.parse_args()
    return a, TrainConfig(**{f: getattr(a, f) for f in asdict(TrainConfig())})


def masked(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Copy of y that keeps only the entries in idx (the others become -1)."""
    z = np.full_like(y, -1)
    z[idx] = y[idx]
    return z


def make_shot_sets(a, ds, tr, seed) -> dict:
    """Identical labelled subsets for every method. For each task and repeat, one random order of the
    training samples of every class (generator seeded by seed, task index and repeat); k labels per class
    are the first min(k, class size) of that order, so the sets are nested in k and small classes
    contribute all their samples once k exceeds their size. --repeats orders for k <= 64, at most 3 for
    k <= 256, 1 above."""
    def reps(k): return a.repeats if k <= 64 else (min(a.repeats, 3) if k <= 256 else 1)
    shot_sets = {}
    for ti, b in enumerate(ds.tasks):
        y = ds.tasks[b]
        pool = tr[y[tr] >= 0]
        classes = np.unique(y[pool])
        orders = []
        for rep in range(a.repeats):   # one generator per repeat, used for the classes in turn
            rng = np.random.default_rng([seed, ti, rep])
            orders.append([rng.permutation(pool[y[pool] == c]) for c in classes])
        for s in a.shots:
            if s == "all":
                shot_sets[b, s] = [pool]
            else:
                k = int(s)
                shot_sets[b, s] = [np.sort(np.concatenate([o[:k] for o in order])) for order in orders[:reps(k)]]
    return shot_sets


def evaluate(a, ds, te, seed, log, shot_sets, b, name, K, extra):
    """SVM on target task b with kernel K for every shot setting; prints the mean score per setting."""
    line = []
    for s in a.shots:
        res = []
        for rep, trk in enumerate(shot_sets[b, s]):
            r = svm_eval(K, ds.tasks[b], trk, te, seed=seed)
            log(seed=seed, task=b, method=name, shots=s, rep=rep, **extra, **r)
            res.append(r.get("auc", r["bacc"]) if a.dataset == "tox21" else r["bacc"])
        line.append(f"{s}:{np.mean(res):.3f}")
    print(f"  {b:12s} {name:16s} " + " ".join(line), flush=True)


def task_nmi(ds, tr, t, b) -> float:
    """Relatedness of task t to target b: NMI of their labels on the training samples."""
    m = tr[(ds.tasks[t][tr] >= 0) & (ds.tasks[b][tr] >= 0)]
    return float(normalized_mutual_info_score(ds.tasks[t][m], ds.tasks[b][m]))


def source_kernel(a, cfg, ds, tr, va, seed, src):
    """Attention kernel (float32) of a network trained on the source task src, and the training info."""
    y = ds.tasks[src]
    model, info = train_model(ds.X, {src: masked(y, tr)}, {src: masked(y, va)}, {src: int(y.max()) + 1}, cfg,
                              seed, verbose=a.verbose)
    K = KK.key_gauss_kernel(*model.keys(torch.tensor(ds.X, device=next(model.parameters()).device)), gamma=a.gamma)
    print(f"  trained on {src}: {info['epochs']} ep in {info['time']:.0f}s, tau {info['tau']:.2f}", flush=True)
    return K, info


def run_attention(a, cfg, ds, tr, va, te, seed, log):
    """K1 / K2 learned on the two tasks most related to each target. A source network is trained once,
    reused by later targets and freed after its last use."""
    tasks = list(ds.tasks)
    target = partial(evaluate, a, ds, te, seed, log, make_shot_sets(a, ds, tr, seed))
    feats = {"rbf": KK.feature_kernel(ds.X, "rbf"), "linear": KK.feature_kernel(ds.X, "linear")}
    plan = []
    for b in a.targets or tasks:
        ranked = sorted((t for t in tasks if t != b), key=lambda t: -task_nmi(ds, tr, t, b))
        plan.append((b, ranked[:2], [task_nmi(ds, tr, t, b) for t in ranked[:2]]))
    uses = Counter(t for _, srcs, _ in plan for t in srcs)
    cache = {}

    for b, (s1, s2), nmis in plan:
        for src in (s1, s2):
            if src not in cache:
                cache[src], info = source_kernel(a, cfg, ds, tr, va, seed, src)
                log(seed=seed, task=None, method="train", sources=[src], **info)
        K1, K2 = cache[s1], cache[s2]
        target(b, "K1", K1, {"sources": [s1], "nmi": nmis[:1]})
        target(b, "K1+feat", KK.with_features([K1], feats), {"sources": [s1], "nmi": nmis[:1]})
        target(b, "K1+K2", KK.mix(K1, K2, KK.ALPHAS_SUM), {"sources": [s1, s2], "nmi": nmis})
        target(b, "K1+K2+feat", KK.with_features([K1, K2], feats), {"sources": [s1, s2], "nmi": nmis})
        del K1, K2
        for src in (s1, s2):
            uses[src] -= 1
            if uses[src] == 0:
                del cache[src]


def run_features(a, cfg, ds, tr, va, te, seed, log):
    """Feature-kernel baselines on every target task."""
    target = partial(evaluate, a, ds, te, seed, log, make_shot_sets(a, ds, tr, seed))
    Kl = KK.feature_kernel(ds.X, "linear")
    Krs = KK.feature_kernels(ds.X, "rbf", KK.RBF_SCALES)
    fams = {"feat_linear": Kl, "feat_rbf": Krs, "feat_linear+rbf": KK.mix(Kl, Krs[0], KK.ALPHAS)}
    for b in a.targets or list(ds.tasks):
        for name, K in fams.items():
            target(b, name, K, {"sources": []})


def completed_seeds(fn: str) -> set[int]:
    """Seeds already finished in fn, listed in <name>.done."""
    done_fn = fn[: -len(".jsonl")] + ".done"
    return {int(line) for line in open(done_fn) if line.strip()} if os.path.exists(done_fn) else set()


def set_aside_partial(fn: str, seeds: set[int]):
    """Moves the rows of interrupted seeds (and truncated lines) from fn to <name>.jsonl.partial,
    so that rerunning those seeds does not duplicate rows."""
    if not os.path.exists(fn):
        return
    keep, moved = [], []
    for line in open(fn):
        try:
            seed = json.loads(line).get("seed")
        except ValueError:
            seed = None
        (moved if seed is None or seed in seeds else keep).append(line)
    if moved:
        with open(fn + ".partial", "a") as f:   # not *.jsonl, so the figure scripts ignore it
            f.writelines(moved)
        with open(fn + ".tmp", "w") as f:
            f.writelines(keep)
        os.replace(fn + ".tmp", fn)
        print(f"{fn}: {len(moved)} rows of interrupted seeds moved to {fn}.partial", flush=True)


def main():
    a, cfg = parse()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.dataset}_{a.mode}{('_' + a.tag) if a.tag else ''}.jsonl")
    lock = open(fn[: -len(".jsonl")] + ".lock", "w")   # held until the process ends
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{fn}: running in another process, skipping", flush=True)
        return
    todo = [s for s in a.seeds if s not in completed_seeds(fn)]
    if not todo:
        print(f"{fn}: seeds {a.seeds} already done, skipping", flush=True)
        return
    set_aside_partial(fn, set(todo))

    with open(fn, "a") as fout, open(fn[: -len(".jsonl")] + ".done", "a") as fdone:
        def log(**row):
            row.update(dataset=a.dataset, mode=a.mode, n_samples=a.n_samples, gamma=a.gamma,
                       cfg=asdict(cfg), time=time.strftime("%Y-%m-%d %H:%M:%S"))
            fout.write(json.dumps(row) + "\n")
            fout.flush()

        for seed in todo:
            ds = D.load(a.dataset, a.n_samples, seed=seed)
            sp = random_split(next(iter(ds.tasks.values())), ds.n, (a.train_frac, a.val_frac), seed)
            tr, va, te = sp["train"], sp["val"], sp["test"]
            print(f"[{ds.name}] seed {seed}  N={ds.n} d={ds.X.shape[1]} train/val/test={len(tr)}/{len(va)}/{len(te)}",
                  flush=True)
            (run_attention if a.mode == "attention" else run_features)(a, cfg, ds, tr, va, te, seed, log)
            fdone.write(f"{seed}\n")   # marks the seed as complete
            fdone.flush()
    print("results ->", fn)


if __name__ == "__main__":
    main()
