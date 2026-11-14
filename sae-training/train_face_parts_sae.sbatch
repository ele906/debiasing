#!/bin/bash
#SBATCH --job-name=sae_face_parts
#SBATCH --partition=all
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face_parts_sae-%j.out

set -euo pipefail
cd /n/fs/goose

/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/fine_tune_face_parts_sae.py \
  --epochs 20 \
  --snapshot-interval 4 \
  --output sae-training/runs/face_parts_sae_001
