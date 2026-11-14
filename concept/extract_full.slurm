#!/bin/bash
#SBATCH --job-name=concept_full
#SBATCH --partition=all
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/n/fs/goose/concept/logs/%x-%A_%a.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/n/fs/goose/concept/deps${PYTHONPATH:+:$PYTHONPATH}
export HF_MODULES_CACHE=/n/fs/goose/concept/hf_modules_${SLURM_ARRAY_TASK_ID}
read -r CONCEPT_REVISION < concept/model_revision.txt
CONCEPT_SHARD=/n/fs/goose/concept/full_run/shard_${SLURM_ARRAY_TASK_ID}
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/extract.py --review "$CONCEPT_SHARD/input.csv" --output "$CONCEPT_SHARD" --summary-dir "$CONCEPT_SHARD" --include-test --revision "$CONCEPT_REVISION"
