#!/bin/bash
#SBATCH --job-name=audit_fairface
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu
#SBATCH --dependency=afterok:3745426

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO
PYTHONPATH=. python ../FairFace/scripts/audit_fairface_independent.py \
  --data_dir /n/fs/goose/data/21082026/two_person_fairness_v2 \
  --fairface_weights fairface_weights/res34_fair_align_multi_7_20190809.pt \
  --cache_dir /n/fs/goose/el8403/hf-cache
