#!/bin/bash
#SBATCH --job-name=mid0-fixed4228
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/n/fs/goose/experiments/mid0_sae_selectivity/fixed4228-%A_%a.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
PY=/n/fs/goose/el8403/conda-envs/goose/bin/python
WORKER=$($PY -c 'import json,sys,os;print(json.load(open(sys.argv[1]+"/config/parallel_workers.json"))[int(os.environ["SLURM_ARRAY_TASK_ID"])]["run"])' "$1")
$PY -u experiments/mid0_sae_selectivity/additive.py --run "$WORKER"
