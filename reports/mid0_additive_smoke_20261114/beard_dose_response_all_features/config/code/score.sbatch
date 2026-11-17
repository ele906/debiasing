#!/bin/bash
#SBATCH --job-name=beard-existing
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/goose/instance-segmentation/beard-existing-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/experiments/mid0_sae_selectivity/deps
run_dir="$1"
shift
/n/fs/goose/el8403/conda-envs/goose/bin/python -u "$run_dir/config/code/infer.py" --run "$run_dir" "$@"
