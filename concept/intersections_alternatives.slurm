#!/bin/bash
#SBATCH --job-name=concept_intersection_alts
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/n/fs/goose/concept/logs/%x-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/intersections.py --all-top-candidates --concepts necklace afro beard --output concept/intersections_alternatives
