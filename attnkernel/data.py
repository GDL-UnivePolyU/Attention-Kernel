"""Dataset loaders: AwA2, CelebA and Tox21.
"""
from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.io import loadmat

from .download import ensure_awa2, ensure_celeba, ensure_tox21

DATA_ROOT = os.environ.get("AK_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # (N, d) float32
    tasks: dict[str, np.ndarray]       # task name -> (N,) int labels, -1 = no label
    class_names: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.X.shape[0]


def subsample(ds: Dataset, n: int | None, seed: int) -> Dataset:
    """n samples drawn at random with the given seed (all samples if n is None or >= N)."""
    if n is None or n >= ds.n:
        return ds
    idx = np.sort(np.random.default_rng(seed).choice(ds.n, n, replace=False))
    return Dataset(ds.name, ds.X[idx], {k: v[idx] for k, v in ds.tasks.items()}, ds.class_names)


# --------------------------------------------------------------------------- AwA2
# The 85 AwA predicates in the order of att_splits.mat (Xian et al., xlsa17); predicates.txt of the
# AwA2 base package is used instead when present.
AWA_PREDICATES = (
    "black white blue brown gray orange red yellow patches spots stripes furry hairless "
    "toughskin big small bulbous lean flippers hands hooves pads paws longleg longneck tail "
    "chewteeth meatteeth buckteeth strainteeth horns claws tusks smelly flys hops swims "
    "tunnels walks fast slow strong weak muscle bipedal quadrapedal active inactive nocturnal "
    "hibernate agility fish meat plankton vegetation insects forager grazer hunter scavenger "
    "skimmer stalker newworld oldworld arctic coastal desert bush plains forest fields jungle "
    "mountains ocean ground water tree cave fierce timid smart group solitary nestspot domestic"
).split()
AWA_TASKS = {
    "habitat": ["ocean", "coastal", "desert", "forest", "plains", "arctic"],
    "diet": ["meat", "fish", "vegetation", "insects", "plankton"],
    "movement": ["swims", "flys", "walks", "hops", "tunnels"],
}


def load_awa2() -> Dataset:
    """AwA2 from the xlsa17 package (res101.mat: ResNet-101 features; att_splits.mat: attribute
    strengths per species). Each task is a set of attributes; an image is labelled with the
    attribute of the set that has the highest strength for its species."""
    root = os.path.join(DATA_ROOT, "xlsa17", "data", "AWA2")
    ensure_awa2(root, os.path.join(DATA_ROOT, "download"))
    res = loadmat(os.path.join(root, "res101.mat"))
    att = loadmat(os.path.join(root, "att_splits.mat"))
    X = res["features"].T.astype(np.float32)
    species = res["labels"].ravel().astype(int) - 1
    A = att["original_att"].T if "original_att" in att else att["att"].T     # (50, 85) strengths
    pred_file = os.path.join(root, "predicates.txt")
    names = [l.split()[1] for l in open(pred_file)] if os.path.exists(pred_file) else AWA_PREDICATES
    tasks, cn = {}, {}
    for t, cls in AWA_TASKS.items():
        s = A[:, [names.index(c) for c in cls]]
        lab = np.where(s.max(1) > 0, s.argmax(1), -1)
        tasks[t], cn[t] = lab[species], cls
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return Dataset("AwA2", X, tasks, cn)


# ------------------------------------------------------------------------- CelebA
CELEBA_BINARY = {   # task -> (attribute, inverted)
    "male": ("Male", False),
    "beard": ("No_Beard", True),
    "makeup": ("Heavy_Makeup", False),
    "smiling": ("Smiling", False),
    "young": ("Young", False),
}
CELEBA_SETS = {   # one attribute out of a set; labelled only if exactly one attribute is present
    "hair_colour": ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"],
    "hair_style": ["Straight_Hair", "Wavy_Hair", "Bald"],
}


def load_celeba() -> Dataset:
    """CelebA (all 202,599 images) from frozen ImageNet ResNet-18 features, extracted on first use
    (download.ensure_celeba): an .npz with X (N, 512), attr (N, 40) in {0, 1} and attr_names (40,)."""
    path = os.path.join(DATA_ROOT, "celeba_full_resnet18.npz")
    ensure_celeba(path, os.path.join(DATA_ROOT, "download"))
    z = np.load(path, allow_pickle=True)
    X, attr, names = z["X"].astype(np.float32), z["attr"], list(z["attr_names"])
    tasks, cn = {}, {}
    for t, (a, inv) in CELEBA_BINARY.items():
        v = attr[:, names.index(a)] > 0
        tasks[t], cn[t] = (~v if inv else v).astype(int), [f"not {t}", t]
    for t, cls in CELEBA_SETS.items():
        a = attr[:, [names.index(c) for c in cls]] > 0
        tasks[t], cn[t] = np.where(a.sum(1) == 1, a.argmax(1), -1), cls
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return Dataset("CelebA", X, tasks, cn)


# -------------------------------------------------------------------------- Tox21
TOX21_TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
               "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


def load_tox21(radius: int = 2, bits: int = 2048) -> Dataset:
    """Tox21 (MoleculeNet): 12 binary assays, missing labels as -1; features are 2,048-bit Morgan
    fingerprints of radius 2 (molecules that RDKit cannot parse are dropped)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    path = os.path.join(DATA_ROOT, "download", "tox21", "tox21.csv.gz")
    ensure_tox21(path)
    rows = list(csv.DictReader(gzip.open(path, "rt")))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    X, Y = [], []
    for r in rows:
        m = Chem.MolFromSmiles(r["smiles"])
        if m is None:
            continue
        X.append(gen.GetFingerprintAsNumPy(m).astype(np.float32))
        Y.append([int(float(r[t])) if r[t] != "" else -1 for t in TOX21_TASKS])
    X, Y = np.stack(X), np.array(Y)
    return Dataset("Tox21", X, {t: Y[:, i] for i, t in enumerate(TOX21_TASKS)},
                   {t: ["inactive", "active"] for t in TOX21_TASKS})


def load(name: str, n_samples: int | None = None, seed: int = 0) -> Dataset:
    ds = {"awa2": load_awa2, "celeba": load_celeba, "tox21": load_tox21}[name.lower()]()
    return subsample(ds, n_samples, seed)
