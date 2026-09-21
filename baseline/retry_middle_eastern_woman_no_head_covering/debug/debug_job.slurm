#!/bin/sh
#SBATCH --job-name=debug_load
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/n/fs/goose/baseline/logs/%x-%j.out

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

python -u /n/fs/goose/baseline/retry_middle_eastern_woman_no_head_covering/debug/debug_load.py
