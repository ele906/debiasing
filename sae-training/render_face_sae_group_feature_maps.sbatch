#!/bin/bash
#SBATCH --job-name=face_sae_group_maps
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=00:30:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face-sae-group-maps-%j.out

set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/n/fs/goose/el8403/conda-envs/reno/bin/python -u \
  sae-training/render_face_sae_group_feature_maps.py
