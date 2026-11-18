#!/bin/bash
#SBATCH --job-name=up000-sae-sweep
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=08:00:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/sweep-%A_%a.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
mode=${1:?Specify pilot or full}
exec el8403/conda-envs/goose/bin/python -u experiments/up000_sae/sweep.py "$mode" "$SLURM_ARRAY_TASK_ID"
