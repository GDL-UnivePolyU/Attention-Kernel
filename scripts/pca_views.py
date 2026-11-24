import argparse
import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from attnkernel import data as D  # noqa: E402
from attnkernel import kernels as KK  # noqa: E402
from attnkernel.train import TrainConfig, random_split, train_model  # noqa: E402


def kpca(K: np.ndarray, k: int = 3) -> np.ndarray:
    """Top-k kernel principal components (centred kernel), one row per sample."""
    K = K.astype(np.float64)
    m = K.mean(0)
    Kc = K - m[None, :] - m[:, None] + m.mean()
    w, V = eigsh(Kc, k=k, which="LA")
    o = np.argsort(w)[::-1]
    return V[:, o] * np.sqrt(np.maximum(w[o], 0))


def masked(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    z = np.full_like(y, -1)
    z[idx] = y[idx]
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["awa2", "celeba", "tox21"])
    p.add_argument("--target", required=True)
    p.add_argument("--sources", nargs=2, required=True, help="most related source, second source")
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=KK.GAMMA)
    p.add_argument("--subsample", type=int, default=0, help="plot a random subset of the test samples")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    cfg = TrainConfig(device=a.device)

    ds = D.load(a.dataset, a.n_samples, seed=a.seed)
    sp = random_split(next(iter(ds.tasks.values())), ds.n, (0.6, 0.15), a.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Ks = []
    for s in a.sources:
        y = ds.tasks[s]
        model, _ = train_model(ds.X, {s: masked(y, tr)}, {s: masked(y, va)}, {s: int(y.max()) + 1}, cfg, a.seed)
        K = KK.key_gauss_kernel(*model.keys(torch.tensor(ds.X, device=next(model.parameters()).device)),
                                gamma=a.gamma)
        Ks.append(K / KK.fro(K))
    Xc = ds.X.astype(np.float64) - ds.X.mean(0)
    U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    views = {"feat": U[:, :3] * sv[:3], "one": kpca(Ks[0]), "two": kpca(Ks[0] + Ks[1])}

    y = ds.tasks[a.target]
    m = te[y[te] >= 0]
    if a.subsample and a.subsample < len(m):
        m = np.sort(np.random.default_rng(0).choice(m, a.subsample, replace=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for name, Z in views.items():
        Z = Z[m] / np.abs(Z[m]).max(0)
        np.savetxt(f"{a.out}_{name}.dat", np.c_[Z, y[m]], fmt=["%.4f"] * 3 + ["%d"], header="x y z c", comments="")
        print("wrote", f"{a.out}_{name}.dat")


if __name__ == "__main__":
    main()
