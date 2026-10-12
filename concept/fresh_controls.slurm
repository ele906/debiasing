#!/bin/bash
#SBATCH --job-name=concept_fresh_controls
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
export PYTHONPATH=/n/fs/goose/concept/deps${PYTHONPATH:+:$PYTHONPATH}
export HF_MODULES_CACHE=/n/fs/goose/concept/hf_modules
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/fresh_controls.py
read -r CONCEPT_REVISION < concept/model_revision.txt
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/extract.py --review concept/independent_controls/manifest.csv --output concept/independent_controls/detector --summary-dir concept/independent_controls/detector --revision "$CONCEPT_REVISION"
