#!/bin/bash
#SBATCH --job-name=two_person_check
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
PYTHONPATH=. python scripts/racial_composition_generate.py \
  --model sdxl-turbo \
  --cache_dir /n/fs/goose/el8403/hf-cache \
  --save_dir /n/fs/goose/data/21082026/two_person_check \
  --n_iters 1 \
  --prompts \
    "A photo of two nurses talking in a hallway" \
    "A photo of two software engineers pairing at a desk" \
    "A photo of two CEOs in a meeting" \
  --n_seeds 20
