#!/bin/bash
#SBATCH --job-name=up000-pilot-check
#SBATCH --partition=all
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/validate-%j.out
set -euo pipefail
cd /n/fs/goose
exec el8403/conda-envs/goose/bin/python experiments/up000_sae/validate_pilot.py
