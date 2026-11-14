#!/bin/bash
#SBATCH --job-name=sae_parts_test2_eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=08:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face_parts_test2_eval-%j.out

set -euo pipefail
DATA=/n/fs/goose/sae-training/runs/face_parts_sae_001/independent_test_seed200_299
RUN=/n/fs/goose/sae-training/runs/face_parts_sae_001
cd "$DATA"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/n/fs/goose/concept/deps${PYTHONPATH:+:$PYTHONPATH}
export HF_MODULES_CACHE=/n/fs/goose/concept/hf_modules_independent_test_200_299
read -r CONCEPT_REVISION < /n/fs/goose/concept/model_revision.txt

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/concept/extract.py \
  --review "$DATA/manifest.csv" \
  --output "$DATA/detector" \
  --summary-dir "$DATA/detector" \
  --revision "$CONCEPT_REVISION" \
  --include-test

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/concept/category_counts.py \
  --predictions "$DATA/detector/concept_predictions.csv" \
  --records-dir "$DATA/detector" \
  --output "$DATA/annotations"

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/sae-training/evaluate_face_parts_extension.py \
  --data "$DATA" \
  --run "$RUN"

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/sae-training/render_face_region_heatmaps.py \
  "$RUN" \
  --data "$DATA" \
  --boxes "$DATA/annotations/bounding_boxes.csv" \
  --residuals "$DATA/residuals" \
  --output "$DATA/region_heatmaps"
