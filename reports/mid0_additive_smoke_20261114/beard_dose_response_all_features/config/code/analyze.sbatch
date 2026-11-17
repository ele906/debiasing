#!/bin/bash
#SBATCH --job-name=beard-plots
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/instance-segmentation/beard-plots-%j.out
set -euo pipefail
cd /n/fs/goose
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps
/n/fs/goose/el8403/conda-envs/goose/bin/python -u "$1/config/code/analyze.py" --run "$1"
