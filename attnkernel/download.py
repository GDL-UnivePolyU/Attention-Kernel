"""Automatic download of AwA2, CelebA and Tox21 into the paths read by data.py.
"""
from __future__ import annotations

import fcntl
import io
import os
import urllib.request
import zipfile
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import models, transforms

AWA2_URL = "https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
AWA2_FILES = ("res101.mat", "att_splits.mat")
TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
# Hugging Face mirror of CelebA: 25 parquet shards (19 train, 3 valid, 3 test; ~12 GB), 202,599 images
CELEBA_BASE_URL = "https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/img_align+identity+attr/"
CELEBA_FILES = ([f"train-{i:05d}-of-00019.parquet" for i in range(19)]
                + [f"{s}-{i:05d}-of-00003.parquet" for s in ("valid", "test") for i in range(3)])


@contextmanager
def locked(path: str) -> Iterator[None]:
    """Exclusive lock on path + '.lock' (blocks until other processes release it)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def fetch(url: str, dest: str):
    """Download url to dest through a temporary file, so that dest is never left incomplete."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url}\n  -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest + ".part")
    os.replace(dest + ".part", dest)


def ensure_awa2(root: str, download_dir: str):
    """res101.mat and att_splits.mat of AwA2, extracted from the xlsa17 archive (~920 MB)."""
    with locked(os.path.join(root, "awa2")):
        if all(os.path.exists(os.path.join(root, f)) for f in AWA2_FILES):
            return
        archive = os.path.join(download_dir, "xlsa17.zip")
        if not os.path.exists(archive):
            fetch(AWA2_URL, archive)
        with zipfile.ZipFile(archive) as z:
            for f in AWA2_FILES:
                dest = os.path.join(root, f)
                with z.open(f"xlsa17/data/AWA2/{f}") as src, open(dest + ".part", "wb") as dst:
                    dst.write(src.read())
                os.replace(dest + ".part", dest)
        os.remove(archive)   # only the two AwA2 files are needed


def ensure_tox21(path: str):
    with locked(path):
        if not os.path.exists(path):
            fetch(TOX21_URL, path)


def ensure_celeba(path: str, download_dir: str):
    """ResNet-18 features of all CelebA images (25 shards), written to path (.npz).

    Restartable: every shard is downloaded and turned into its own feature file (download_dir/celeba_parts),
    and finished shards are skipped. The parts are merged in file-name order (test, train, valid)."""
    with locked(path):
        if os.path.exists(path):
            return
        parts_dir = os.path.join(download_dir, "celeba_parts")
        for f in CELEBA_FILES:
            part = os.path.join(parts_dir, f.replace(".parquet", ".npz"))
            if os.path.exists(part):
                continue
            shard = os.path.join(download_dir, "celeba", f)
            if not os.path.exists(shard):
                fetch(CELEBA_BASE_URL + f, shard)
            extract_celeba_features([shard], part)
        parts = [np.load(os.path.join(parts_dir, f.replace(".parquet", ".npz")), allow_pickle=True)
                 for f in sorted(CELEBA_FILES)]
        names = parts[0]["attr_names"]
        assert all((p["attr_names"] == names).all() for p in parts), "attribute order differs between shards"
        with open(path + ".part", "wb") as fh:   # file handle: numpy would append .npz to a .part name
            np.savez_compressed(fh, X=np.concatenate([p["X"] for p in parts]).astype(np.float32),
                                attr=np.concatenate([p["attr"] for p in parts]), attr_names=names,
                                celeb_id=np.concatenate([p["celeb_id"] for p in parts]))
        os.replace(path + ".part", path)
        print(f"{len(parts)} shards -> {path}", flush=True)


def extract_celeba_features(shards: list[str], out: str, size: int = 224, batch: int = 128):
    """Frozen ImageNet ResNet-18 features (512-d, centre crop) of CelebA parquet shards (Hugging Face
    mirror flwrlabs/celeba, config img_align+identity+attr). Writes X (N, 512), attr (N, 40) in {0,1},
    attr_names (40,) and celeb_id (N,)."""
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    feats, attrs, ids, names = [], [], [], None
    for shard in shards:
        t = pq.read_table(shard)
        names = [c for c in t.column_names if c not in ("image", "celeb_id")]
        attrs.append(np.stack([t[c].to_numpy(zero_copy_only=False) for c in names], 1).astype(np.int8))
        ids.append(t["celeb_id"].to_numpy())
        imgs = t["image"].to_pylist()
        for s in range(0, len(imgs), batch):
            x = torch.stack([tf(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in imgs[s:s + batch]])
            with torch.no_grad():
                feats.append(net(x.to(dev)).cpu().numpy())
            print(f"{shard}: {min(s + batch, len(imgs))}/{len(imgs)}", end="\r", flush=True)
        print()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out + ".part", "wb") as f:   # file handle: numpy would append .npz to a .part name
        np.savez_compressed(f, X=np.concatenate(feats).astype(np.float32), attr=np.concatenate(attrs),
                            attr_names=np.array(names), celeb_id=np.concatenate(ids))
    os.replace(out + ".part", out)
