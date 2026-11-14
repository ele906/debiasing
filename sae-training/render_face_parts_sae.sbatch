#!/bin/bash
#SBATCH --job-name=sae_parts_maps
#SBATCH --partition=all
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face_parts_maps-%j.out

set -euo pipefail
cd /n/fs/goose

/n/fs/goose/el8403/conda-envs/reno/bin/python \
  sae-training/render_face_region_heatmaps.py \
  sae-training/runs/face_parts_sae_001
