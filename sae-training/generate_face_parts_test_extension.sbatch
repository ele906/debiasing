#!/bin/bash
#SBATCH --job-name=sae_parts_test_images
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face_parts_test_images-%j.out

set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/n/fs/goose/el8403/conda-envs/reno/bin/python -u backprop/generate_images.py \
  --count 100 \
  --seed-start 100 \
  --test-only \
  --save-residuals \
  --output sae-training/runs/face_parts_sae_001/test_extension
