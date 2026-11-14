#!/bin/bash
#SBATCH --job-name=mid0-spatial-eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/experiments/mid0_sae_selectivity/spatial-eval-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH= /usr/local/anaconda3/2024.02/bin/python -u experiments/mid0_sae_selectivity/spatial_code/face_evaluate.py --run "$1"
PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps /n/fs/goose/el8403/conda-envs/goose/bin/python -u experiments/mid0_sae_selectivity/spatial_code/evaluate.py --run "$1"
