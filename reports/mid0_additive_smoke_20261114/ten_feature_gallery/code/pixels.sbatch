#!/bin/bash
#SBATCH --job-name=old-photo-metrics
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/n/fs/goose/experiments/mid0_sae_selectivity/old-photo-metrics-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps
/n/fs/goose/el8403/conda-envs/goose/bin/python -u experiments/mid0_sae_selectivity/ten_feature_code/pixel_metrics.py --run "$1"
