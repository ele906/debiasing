#!/bin/bash
#SBATCH --job-name=two_person_fairness_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
# Rerun (not reusing outputs/two_person_fairness) so rewards.csv gets
# written per-iteration by the updated training/trainer.py -- job 3745422
# already had the old trainer.py loaded into its Python process before that
# change landed, so its output dir won't have rewards.csv for the
# quality-vs-fairness tradeoff plot.
PYTHONPATH=. python scripts/racial_composition_generate.py \
  --model sdxl-turbo \
  --cache_dir /n/fs/goose/el8403/hf-cache \
  --save_dir /n/fs/goose/data/21082026/two_person_fairness_v2 \
  --prompts \
    "A photo of two nurses talking in a hallway" \
    "A photo of two software engineers pairing at a desk" \
    "A photo of two CEOs in a meeting" \
  --n_seeds 20 \
  --enable_fairness \
  --fairness_weighting 3.0 \
  --fairface_weights fairface_weights/res34_fair_align_multi_7_20190809.pt

PYTHONPATH=. python scripts/racial_composition_analyze_by_stage.py \
  --data_dir /n/fs/goose/data/21082026/two_person_fairness_v2 \
  --weights fairface_weights/res34_fair_align_multi_7_20190809.pt
