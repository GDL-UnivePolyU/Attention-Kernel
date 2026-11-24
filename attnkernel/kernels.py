"""Kernels over all N samples: the attention kernel, the feature kernels and their combinations.
"""
from __future__ import annotations

import numpy as np
import torch

GAMMA = 0.25                           # bandwidth factor of the attention kernel
ROW_CHUNK = 2048
MEDIAN_MAX_N = 20000                   # median heuristic on a random subset above this size
ALPHAS = (0.5, 0.25, 0.75, 0.1, 0.9)   # weight of the first kernel in kernel + feature kernel (first = default)
ALPHAS_SUM = (0.5, 0.25, 0.75)         # weight of K1 in the sum of two attention kernels
RBF_SCALES = (1.0, 0.25, 4.0)          # RBF baseline: median heuristic times these (first = default)


# ------------------------------------------------------------------ attention kernel
@torch.no_grad()
def key_gauss_kernel(keys: torch.Tensor, tau, gamma: float = GAMMA, chunk: int = ROW_CHUNK) -> np.ndarray:
    """Attention kernel from the keys (h, N, dh) and the temperature (model.keys), built in row chunks on
    the keys' device without forming the (h, N, N) logits: mean over heads of
    exp(gamma (s_ij - (s_ii + s_jj)/2)), float32 (N, N), unit diagonal."""
    n = keys.shape[1]
    diag = tau * (keys * keys).sum(-1)                                # s_ii, (h, N)
    keys_t = keys.transpose(1, 2).contiguous()                        # contiguous: wrong matmul on MPS otherwise
    K = np.empty((n, n), np.float32)
    for s in range(0, n, chunk):
        r = slice(s, min(n, s + chunk))
        S = tau * keys[:, r].contiguous() @ keys_t                    # (h, rows, N)
        L = (S - 0.5 * (diag[:, r, None] + diag[:, None, :])).clamp_max(0.0)
        K[r] = torch.exp(gamma * L).mean(0).float().cpu().numpy()
    symmetrise(K, chunk)
    np.fill_diagonal(K, 1.0)
    return K


def symmetrise(K: np.ndarray, chunk: int = ROW_CHUNK):
    """K <- (K + K^T)/2 in place, block by block (no N x N temporary)."""
    n = len(K)
    for i in range(0, n, chunk):
        for j in range(i, n, chunk):
            a, b = slice(i, min(n, i + chunk)), slice(j, min(n, j + chunk))
            m = 0.5 * (K[a, b] + K[b, a].T)
            K[a, b], K[b, a] = m, m.T


# ------------------------------------------------------------------- feature kernels
def _median_sq_dist(X: np.ndarray, max_n: int = MEDIAN_MAX_N, seed: int = 0) -> float:
    """Median of the non-zero squared distances; exact up to max_n samples, else on a random subset."""
    if len(X) > max_n:
        X = X[np.random.default_rng(seed).choice(len(X), max_n, replace=False)]
    sq = (X ** 2).sum(1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
    return float(np.median(D[D > 0]))


def feature_kernels(X: np.ndarray, kind: str = "linear", scales=(1.0,), chunk: int = ROW_CHUNK) -> list[np.ndarray]:
    """Linear kernel, or RBF kernels exp(-scale ||x - y||^2 / median) for each scale (median heuristic);
    float32, unit mean diagonal, built in row chunks."""
    if kind not in ("linear", "rbf"):
        raise ValueError(kind)
    X = X.astype(np.float64)
    n = len(X)
    scales = (1.0,) if kind == "linear" else scales
    out = [np.empty((n, n), np.float32) for _ in scales]
    sq = (X ** 2).sum(1)
    med = _median_sq_dist(X) if kind == "rbf" else None
    for s in range(0, n, chunk):
        r = slice(s, min(n, s + chunk))
        G = X[r] @ X.T
        if kind == "rbf":
            G = np.maximum(sq[r, None] + sq[None, :] - 2 * G, 0)   # squared distances
        for K, sc in zip(out, scales):
            K[r] = G if kind == "linear" else np.exp(-sc / med * G)
    for K in out:
        K /= np.float32(np.mean(np.diag(K), dtype=np.float64))
    return out


def feature_kernel(X: np.ndarray, kind: str = "linear", scale: float = 1.0) -> np.ndarray:
    """One feature kernel (see feature_kernels)."""
    return feature_kernels(X, kind, (scale,))[0]


# ------------------------------------------------------------------------ candidates
def fro(A: np.ndarray, B: np.ndarray | None = None, chunk: int = ROW_CHUNK) -> float:
    """<A, B>_F (or ||A||_F if B is None), accumulated in float64 by row chunks."""
    acc = 0.0
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk].astype(np.float64)
        acc += float(np.sum(a * a)) if B is None else float(np.sum(a * B[s:s + chunk]))
    return float(np.sqrt(acc)) if B is None else acc


class Candidates:
    """Candidate kernels sum_k c_k B_k over shared base kernels B_k, never formed in full: `block` builds
    only the rows x cols an SVM needs. With unit_diag every candidate is rescaled to unit mean diagonal."""

    def __init__(self, bases: list[np.ndarray], coefs, unit_diag: bool = True):
        self.bases, self.coefs = list(bases), [np.asarray(c, np.float64) for c in coefs]
        dmeans = np.array([np.mean(np.diag(B), dtype=np.float64) for B in self.bases])
        self.scale = [1.0 / float(c @ dmeans) if unit_diag else 1.0 for c in self.coefs]

    def __len__(self):
        return len(self.coefs)

    def block(self, i: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        ix = np.ix_(rows, cols)
        terms = [(c, B) for c, B in zip(self.coefs[i], self.bases) if c != 0]
        if len(terms) == 1 and terms[0][0] * self.scale[i] == 1.0:   # a base kernel as it is
            return terms[0][1][ix]
        out = sum(c * B[ix].astype(np.float64) for c, B in terms)
        return (out * self.scale[i]).astype(np.float32)


def as_candidates(K) -> Candidates:
    """Candidates as they are; a list of kernels (first = default) or a single kernel, unscaled."""
    if isinstance(K, Candidates):
        return K
    Ks = list(K) if isinstance(K, (list, tuple)) else [K]
    return Candidates(Ks, np.eye(len(Ks)), unit_diag=False)


def mix(Ka: np.ndarray, Kb: np.ndarray, alphas=ALPHAS_SUM) -> Candidates:
    """alpha Ka/||Ka||_F + (1 - alpha) Kb/||Kb||_F for each alpha, unit mean diagonal."""
    na, nb = fro(Ka), fro(Kb)
    return Candidates([Ka, Kb], [[al / na, (1 - al) / nb] for al in alphas])


def with_features(Ks: list[np.ndarray], feats: dict[str, np.ndarray], alphas=ALPHAS) -> Candidates:
    """K = sum of the Frobenius-normalised kernels in Ks, combined with each feature kernel:
    alpha K/||K||_F + (1 - alpha) Kf/||Kf||_F, candidates over the feature kernels (first = default)
    and alpha (first = default), unit mean diagonal."""
    norms = [fro(K) for K in Ks]
    # ||sum_i K_i/||K_i|| ||_F from the inner products of the K_i
    nsum = np.sqrt(sum(fro(Ks[i], Ks[j]) / (norms[i] * norms[j]) for i in range(len(Ks)) for j in range(len(Ks))))
    fk = list(feats.values())
    coefs = []
    for j, Kf in enumerate(fk):
        nf = fro(Kf)
        for al in alphas:
            c = [al / (n * nsum) for n in norms] + [0.0] * len(fk)
            c[len(Ks) + j] = (1 - al) / nf
            coefs.append(c)
    return Candidates(list(Ks) + fk, coefs)
