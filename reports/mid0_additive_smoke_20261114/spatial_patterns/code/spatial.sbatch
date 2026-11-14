#!/bin/bash
#SBATCH --job-name=mid0-spatial4228
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/experiments/mid0_sae_selectivity/spatial-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
PY=/n/fs/goose/el8403/conda-envs/goose/bin/python
if [[ "$2" == smoke ]]; then
  $PY -u "$1/config/source/worker.py" --run "$1" --seed 30 --smoke
else
  SEED=$((30 + SLURM_ARRAY_TASK_ID))
  $PY -u "$1/config/source/worker.py" --run "$1" --seed "$SEED"
fi
