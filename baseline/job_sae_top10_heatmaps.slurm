#!/bin/sh
#SBATCH --job-name=sae_top10_heatmaps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
jupyter nbconvert --to notebook --execute --inplace \
  /n/fs/goose/baseline/sae_top10_heatmaps.ipynb
