#!/bin/bash
#SBATCH --job-name=up000-spatial
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/spatial-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
sha256sum -c experiments/up000_sae/training/checkpoint.sha256
el8403/conda-envs/goose/bin/python -u experiments/up000_sae/spatial.py
