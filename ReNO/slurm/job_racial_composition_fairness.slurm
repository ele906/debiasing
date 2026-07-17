#!/bin/bash
#SBATCH --job-name=racial_composition_fairness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=neu306
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
PYTHONPATH=. python scripts/racial_composition_generate.py \
  --model sdxl-turbo \
  --cache_dir /n/fs/goose/el8403/hf-cache \
  --save_dir outputs/racial_composition_fairness_v2 \
  --prompts "A photo of a CEO" "A photo of a nurse" "A photo of a software engineer" \
  --n_seeds 20 \
  --enable_fairness \
  --fairness_weighting 15.0 \
  --fairface_weights fairface_weights/res34_fair_align_multi_7_20190809.pt

PYTHONPATH=. python scripts/racial_composition_analyze_by_stage.py \
  --data_dir outputs/racial_composition_fairness_v2 \
  --weights fairface_weights/res34_fair_align_multi_7_20190809.pt
