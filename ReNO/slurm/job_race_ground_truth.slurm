#!/bin/bash
#SBATCH --job-name=race_ground_truth_more
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
# 14 categories (7 races x 2 genders) x 60 seeds = 840 plain (non-optimized)
# SDXL-turbo samples, used as a synthetic ground-truth test set for FairFace.
# Bumped from 20->60 seeds for a more accurate per-category accuracy estimate;
# already-generated seed{0..19}.png are skipped (see generate_race_ground_truth.py).
PYTHONPATH=. python ../FairFace/scripts/generate_race_ground_truth.py \
  --model sdxl-turbo \
  --cache_dir /n/fs/goose/el8403/hf-cache \
  --save_dir /n/fs/goose/FairFace/data/ground_truth \
  --n_seeds 60

PYTHONPATH=. python scripts/score_fairface_on_ground_truth.py \
  --data_dir /n/fs/goose/FairFace/data/ground_truth \
  --weights fairface_weights/res34_fair_align_multi_7_20190809.pt
