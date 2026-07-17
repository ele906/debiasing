#!/bin/bash
#SBATCH --job-name=est_bias_dir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
PYTHONPATH=. python scripts/estimate_bias_direction.py \
  --model sdxl-turbo \
  --cache_dir hf_cache \
  --n_seeds 8 \
  --output bias_directions/sdxl-turbo.pt