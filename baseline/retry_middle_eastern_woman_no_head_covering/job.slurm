#!/bin/sh
#SBATCH --job-name=me_woman_no_head_covering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/n/fs/goose/baseline/logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
jupyter nbconvert --to notebook --execute --inplace \
  /n/fs/goose/baseline/retry_middle_eastern_woman_no_head_covering/generate_and_heatmap.ipynb
