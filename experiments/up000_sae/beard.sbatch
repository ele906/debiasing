#!/bin/bash
#SBATCH --job-name=up000-3742-beard
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/beard-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps
exec el8403/conda-envs/goose/bin/python -u experiments/up000_sae/beard.py
