# Attention as a Kernel

[![Docker Pulls](https://img.shields.io/docker/pulls/ripper346/attention-kernel?logo=docker&label=Image%20pulls)](https://hub.docker.com/r/ripper346/attention-kernel)
![Docker Image Size (tag)](https://img.shields.io/docker/image-size/ripper346/attention-kernel/latest?logo=docker&label=AttentionKernel)

Code for the article *Attention as a Kernel: Uncovering Latent Geometry for Few-Shot Transfer*.

A Transformer attention layer runs over **all samples of a dataset** (training, validation and test): every
sample attends to every other sample. The network is trained on a *source task*; its learned attention then
defines a positive semi-definite kernel between all samples, which an SVM uses on a different *target task*.

## Method

1. Each sample `x_i` is mapped to `h_i = φ(x_i)` (linear layer, LayerNorm, GELU, dropout) and to a unit-norm key
   `k_i = W_K h_i / ‖W_K h_i‖`; queries and keys are tied. The attention between samples `i` and `j` is
   `a_ij = exp(τ k_iᵀk_j) / Σ_{l≠i} exp(τ k_iᵀk_l)`, with a learned temperature `τ` shared by 4 heads and no
   self-attention ([attnkernel/model.py](attnkernel/model.py)).
2. The network is trained with cross-entropy on the source labels of the training samples only, attending over
   all samples ([attnkernel/train.py](attnkernel/train.py)).
3. The exponentiated attention scores, normalised to unit diagonal and raised to the power `γ = 1/4`, give the
   attention kernel ([attnkernel/kernels.py](attnkernel/kernels.py)), averaged over the heads:

   ```
   K_ij = ( exp(τ k_iᵀk_j) / sqrt(exp(τ k_iᵀk_i) exp(τ k_jᵀk_j)) )^γ = exp(-(γτ/2) ‖k_i - k_j‖²)
   ```

   a Gaussian kernel on the learned keys, hence PSD.
4. An SVM with the precomputed kernel is trained on the labelled samples of the target task and evaluated on
   the test samples.

## Layout

| file | content |
|---|---|
| [run.py](run.py) | experiments: `--mode attention` (attention kernels) and `--mode features` (baselines) |
| [run_experiments.sh](run_experiments.sh) | every experiment and the data of both figures |
| [scripts/make_curves.py](scripts/make_curves.py) | learning curves (`curve_*.dat`) from `results/*.jsonl` |
| [scripts/pca_views.py](scripts/pca_views.py) | PCA figure data (`pca/*.dat`) |

## Setup

With Docker (CUDA image, GPU 0):

```bash
docker compose up -d
docker exec -it attention_kernel bash
```

`ripper346/attention-kernel:latest` is a build of the [Dockerfile](Dockerfile) pushed to Docker Hub, so compose
only pulls it; compose mounts the project at `/project`.

Without Docker:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The device is chosen automatically (CUDA, then Apple MPS, then CPU); force one with `--device cpu`.

## Data

Datasets are downloaded into `data/` on first use (override the folder with `AK_DATA=...`); parallel runs
wait for a single download.

* **AwA2**: ResNet-101 features (2,048-d) of 37,322 images of 50 species from the xlsa17 package of Xian et al.
  (~0.9 GB). Attributes are given per species with a strength; each task is a set of attributes and an image is
  labelled with the strongest attribute of the set for its species: *habitat* (ocean, coastal, desert, forest,
  plains, arctic), *diet* (meat, fish, vegetation, insects, plankton), *movement* (swims, flies, walks, hops,
  tunnels).
* **CelebA**: all 202,599 images of the Hugging Face mirror `flwrlabs/celeba` (25 parquet files, ~12 GB), frozen
  ImageNet ResNet-18 features (512-d, 224 × 224 centre crop), extracted on first use into
  `data/celeba_full_resnet18.npz`; download and extraction resume from the last finished file. Tasks: *male*,
  *beard* (inverse of No_Beard), *makeup* (Heavy_Makeup), *smiling*, *young*, and *hair colour* (black, blond,
  brown, grey) and *hair style* (straight, wavy, bald), labelled only when exactly one attribute of the set is
  present.
* **Tox21**: 7,823 molecules (the 7,831 of MoleculeNet that RDKit parses), 2,048-bit Morgan fingerprints of
  radius 2, 12 binary assays with missing labels.

## Running

All experiments and the figure data (results in `results/`, figure data in `results/figures`):

```bash
bash run_experiments.sh                       # 8,000 random samples per seed on AwA2 and CelebA (Tox21: all)
JOBS=6 GPUS="0 1" bash run_experiments.sh     # 6 runs at a time on GPUs 0 and 1
cd results/figures && pdflatex ../../scripts/figures/curves.tex \
  && lualatex ../../scripts/figures/pca_fig.tex && lualatex ../../scripts/figures/pca_fig.tex
```

[run_experiments.sh](run_experiments.sh) runs 3 seeds of both modes per dataset as independent processes in a
pool of `JOBS` (default 3, `THREADS` threads each). It is restartable: `run.py` records finished seeds in
`results/<name>.done` and skips them; an interrupted seed is rerun from scratch, with its partial rows moved to
`results/<name>.jsonl.partial`.

A single run:

```bash
python run.py --dataset awa2 --mode attention --n_samples 8000 --seeds 0
python run.py --dataset tox21 --mode features --seeds 0 1 2
```

Results are appended to `results/<dataset>_<mode>[_<tag>].jsonl`, one row per (seed, task, method, labels per
class, repeat), with `bacc`, `acc`, `auc` (binary tasks), the chosen `C` and candidate `kidx`, the sources and
their NMI; rows with `method = "train"` record the training of each source network (epochs, time, final `τ`).
