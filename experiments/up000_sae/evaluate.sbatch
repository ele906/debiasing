#!/bin/bash
#SBATCH --job-name=up000-sae-eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/evaluate-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
exec /usr/local/anaconda3/2024.02/bin/python -u experiments/up000_sae/evaluate.py "${1:?Specify pilot or full}"
