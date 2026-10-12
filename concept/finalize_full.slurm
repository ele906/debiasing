#!/bin/bash
#SBATCH --job-name=concept_full_summary
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/n/fs/goose/concept/logs/%x-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/full_dataset.py merge
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/compare_sae.py --exclude-test --output concept/sae_comparison_full
/n/fs/goose/el8403/conda-envs/reno/bin/python -u concept/category_counts.py --records-dir concept/full_run/records --output concept/category_analysis_full
