#!/bin/bash
#SBATCH --job-name=two_person_baseline_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
PYTHONPATH=. python scripts/racial_composition_analyze_by_stage.py \
  --data_dir /n/fs/goose/data/21082026/two_person_check \
  --weights fairface_weights/res34_fair_align_multi_7_20190809.pt
