"""Transductive training of the attention network and SVM evaluation of (candidate) kernels."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

from .kernels import as_candidates
from .model import AttentionKernelNet


@dataclass
class TrainConfig:
    d: int = 128
    heads: int = 4
    dropout: float = 0.3
    attn_dropout: float = 0.1
    tau0: float = 10.0
    lr: float = 3e-3
    wd: float = 5e-4
    epochs: int = 300
    patience: int = 50
    full_batch: int = 6000      # attention over the full set if N <= full_batch, else random subsets
    batch: int = 2048           # subset size per step when N > full_batch
    chunk: int = 4096           # row chunk for no-grad forward passes over all N samples
    device: str = "auto"


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(X: np.ndarray, train_labels: dict[str, np.ndarray], val_labels: dict[str, np.ndarray],
                n_classes: dict[str, int], cfg: TrainConfig, seed: int = 0, verbose: bool = False):
    """Train on all samples; only entries >= 0 of train_labels enter the loss, entries >= 0 of
    val_labels are used for early stopping (mean head accuracy). Returns (model, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device(cfg.device)
    n = X.shape[0]
    model = AttentionKernelNet(X.shape[1], n_classes, cfg.d, cfg.heads, cfg.dropout,
                               cfg.attn_dropout, cfg.tau0).to(dev)
    Xt = torch.tensor(X, device=dev)
    ytr = {t: torch.tensor(v, device=dev) for t, v in train_labels.items()}
    yva = {t: torch.tensor(v, device=dev) for t, v in val_labels.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    full = n <= cfg.full_batch
    steps = 1 if full else int(np.ceil(n / cfg.batch))
    best, best_state, bad, best_ep = -np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for _ in range(steps):
            idx = None if full else torch.randperm(n, device=dev)[: cfg.batch]
            out = model(Xt if full else Xt[idx])
            loss = 0.0
            for t, y in ytr.items():
                yb = y if full else y[idx]
                m = yb >= 0
                if m.any():
                    loss = loss + F.cross_entropy(out[t][m], yb[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():   # validation over the full set
            out = model(Xt, chunk=cfg.chunk)
            accs = [(out[t][y >= 0].argmax(1) == y[y >= 0]).float().mean().item()
                    for t, y in yva.items() if (y >= 0).any()]
            score = float(np.mean(accs)) if accs else -float(loss)
        if score > best:
            best, best_state, bad, best_ep = score, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= cfg.patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  ep {ep:4d} loss {float(loss):.4f} val {score:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    info = {"epochs": ep + 1, "best_epoch": best_ep, "time": time.time() - t0, "val": best,
            "tau": float(model.attn.tau())}
    return model, info


def random_split(y: np.ndarray, n: int, frac=(0.6, 0.15), seed: int = 0) -> dict[str, np.ndarray]:
    """Random train/val/test split, stratified on y (the first task) where valid."""
    idx = np.arange(n)
    strat = np.where(y >= 0, y, y.max() + 1)
    tr, rest = train_test_split(idx, train_size=frac[0], stratify=strat, random_state=seed)
    va, te = train_test_split(rest, train_size=frac[1] / (1 - frac[0]), stratify=strat[rest],
                              random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


# ---------------------------------------------------------------------- evaluation
def metrics(y_true, y_pred) -> dict:
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "bacc": float(balanced_accuracy_score(y_true, y_pred))}


def svm_eval(K, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
             Cs=(0.01, 0.1, 1, 10, 100), seed: int = 0) -> dict:
    """SVM with a precomputed kernel and class-balanced weights.

    K is a single kernel, a list of candidate kernels or a kernels.Candidates object. The (candidate, C)
    pair is chosen by 3-fold CV on the labelled samples when every class has >= 3 of them; otherwise
    candidate 0 and C = 1 are used. Only the blocks train x train and test x train are formed.
    """
    K = as_candidates(K)
    tr, te = tr[y[tr] >= 0], te[y[te] >= 0]
    ytr = y[tr]
    if len(np.unique(ytr)) < 2:
        return {"acc": float("nan"), "bacc": float("nan"), "C": None, "kidx": None}
    counts = np.bincount(ytr)
    kidx, C = 0, 1.0
    if counts[counts > 0].min() >= 3:
        skf = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, ytr))
        best = (-1.0, 0, 1.0)
        for ki in range(len(K)):
            Ktt = K.block(ki, tr, tr)
            for c in Cs:
                sc = []
                for a, b in skf:
                    m = SVC(C=c, kernel="precomputed", class_weight="balanced")
                    m.fit(Ktt[np.ix_(a, a)], ytr[a])
                    sc.append(balanced_accuracy_score(ytr[b], m.predict(Ktt[np.ix_(b, a)])))
                if np.mean(sc) > best[0]:
                    best = (float(np.mean(sc)), ki, c)
            del Ktt
        _, kidx, C = best
    m = SVC(C=C, kernel="precomputed", class_weight="balanced").fit(K.block(kidx, tr, tr), ytr)
    Kte = K.block(kidx, te, tr)
    r = metrics(y[te], m.predict(Kte))
    if len(m.classes_) == 2 and len(np.unique(y[te])) == 2:   # ROC-AUC for binary tasks
        r["auc"] = float(roc_auc_score(y[te] == m.classes_[1], m.decision_function(Kte)))
    r["C"], r["kidx"] = C, kidx
    return r
