#!/bin/bash
#SBATCH --job-name=concept_extract
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/n/fs/goose/concept/logs/%x-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/n/fs/goose/concept/deps${PYTHONPATH:+:$PYTHONPATH}
export HF_MODULES_CACHE=/n/fs/goose/concept/hf_modules
read -r CONCEPT_REVISION < concept/model_revision.txt
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/extract.py --revision "$CONCEPT_REVISION"
