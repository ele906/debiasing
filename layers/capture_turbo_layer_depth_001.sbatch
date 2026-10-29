#!/bin/bash
#SBATCH --job-name=turbo_layer_depth
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/goose/layers/logs/turbo-layer-depth-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/n/fs/goose/el8403/conda-envs/reno/bin/python -u sae-training/capture_layerwise_face_sae.py \
  --data sae-training/runs/face_parts_sae_001/test_extension \
  --seed-start 100 --seed-count 100 --chunk-size 20 \
  --cache-dir /n/fs/goose/ReNO/hf_cache \
  --output layers/runs/turbo_layer_depth_001
