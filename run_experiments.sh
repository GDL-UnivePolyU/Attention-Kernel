#!/usr/bin/env bash
# All experiments of the article: attention kernels (--mode attention) and feature baselines
# (--mode features) on AwA2, CelebA and Tox21, 3 seeds, then the data of the PCA and learning-curve figures
# in results/figures (compile scripts/figures/*.tex there, see README).
#
#   bash run_experiments.sh                          # the article: 8,000 random samples per seed (3 runs at a time)
#   JOBS=6 GPUS="0 1" bash run_experiments.sh        # 6 runs at a time, spread over GPUs 0 and 1
#   AWA2_N= JOBS=2 bash run_experiments.sh           # all of AwA2 (~40 GB of RAM per run)
#
# AWA2_N / CELEBA_N: random samples per seed (default 8000); empty = all samples (all of CelebA does not fit
# in memory, see README). Tox21 always uses all molecules.
#
# Restartable: run.py skips the seeds listed in results/<name>.done and reruns an interrupted seed from
# scratch; the PCA data are skipped when already written. Runs are independent (one results file each) and
# go through a pool of JOBS processes, spread round-robin over the GPUs in GPUS. THREADS (default 2) sets
# OMP_NUM_THREADS per run; on CPU, results are reproducible only with the same THREADS.
cd "$(dirname "$0")"
PY=${PY:-python}
JOBS=${JOBS:-3}
read -r -a GPU_LIST <<< "${GPUS:-0}"
THREADS=${THREADS:-2}
AWA2_N=${AWA2_N-8000}
CELEBA_N=${CELEBA_N-8000}
FIG=results/figures
export PYTHONUNBUFFERED=1   # run.py output appears as it is printed
mkdir -p results
LAUNCHED=0

hms() { printf '%dh%02dm%02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

# run_one <name> <command...>: runs one command, streams its output to stdout and to results/<name>.log,
# and reports start, end and duration
run_one() {
  local name=$1 t0 rc
  shift
  t0=$(date +%s)
  echo "$(date '+%F %T')  start   $name  (GPU $RUN_GPU)"
  "$@" 2>&1 | tee -a "results/$name.log" | sed -u "s/^/[$name] /"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then
    echo "$(date '+%F %T')  done    $name  in $(hms $(($(date +%s) - t0)))"
  else
    echo "$(date '+%F %T')  FAILED  $name  after $(hms $(($(date +%s) - t0))) (exit code $rc)"
  fi
}

# launch <name> <command...>: waits for a free slot of the pool, then runs the command in the background
# on the next GPU of GPU_LIST
launch() {
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
    wait -n
  done
  local gpu=${GPU_LIST[$((LAUNCHED % ${#GPU_LIST[@]}))]}
  LAUNCHED=$((LAUNCHED + 1))
  local name=$1
  shift
  RUN_GPU=$gpu run_one "$name" env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$THREADS" "$@" &
}

# ntag <n_samples>: tag prefix of the results files (n8000_ or all_)
ntag() { if [ -n "$1" ]; then echo "n${1}_"; else echo "all_"; fi; }

# pca <name> [pca_views.py args]: PCA figure data (seed 0), skipped when already written
pca() {
  local name=$1
  shift
  [ -f "$FIG/pca/${name}_two.dat" ] || launch "pca_$name" "$PY" scripts/pca_views.py --out "$FIG/pca/$name" "$@"
}

echo "$(date '+%F %T')  pool: JOBS=$JOBS GPUS=${GPU_LIST[*]} THREADS=$THREADS PY=$PY"
AWA2_TAG=$(ntag "$AWA2_N")
CELEBA_TAG=$(ntag "$CELEBA_N")
TOX21_TAG=$(ntag "")
for spec in "awa2 $AWA2_TAG $AWA2_N" "celeba $CELEBA_TAG $CELEBA_N" "tox21 $TOX21_TAG"; do
  read -r ds tag n <<< "$spec"
  for mode in features attention; do
    for s in 0 1 2; do
      launch "${ds}_${mode}_${tag}seed$s" \
        "$PY" run.py --dataset "$ds" --mode "$mode" --seeds "$s" --tag "${tag}seed$s" ${n:+--n_samples $n}
    done
  done
done

# PCA figure: CelebA male from makeup (+ beard) on 1,500 random test samples, AwA2 movement from diet
# (+ habitat), Tox21 NR-AR-LBD from NR-AR (+ NR-ER-LBD)
pca male --dataset celeba ${CELEBA_N:+--n_samples $CELEBA_N} --target male --sources makeup beard --subsample 1500
pca movement --dataset awa2 ${AWA2_N:+--n_samples $AWA2_N} --target movement --sources diet habitat
pca arlbd --dataset tox21 --target NR-AR-LBD --sources NR-AR NR-ER-LBD
wait

"$PY" scripts/make_curves.py --results results --out "$FIG" \
  --awa2_tag "$AWA2_TAG" --celeba_tag "$CELEBA_TAG" --tox21_tag "$TOX21_TAG"
echo "all done: $(date '+%F %T')"
