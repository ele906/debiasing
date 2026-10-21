#!/bin/bash
#SBATCH --job-name=sae_parts_test_boxes
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=08:00:00
#SBATCH --output=/n/fs/goose/sae-training/logs/face_parts_test_boxes-%j.out

set -euo pipefail
EXT=/n/fs/goose/sae-training/runs/face_parts_sae_001/test_extension
cd "$EXT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/n/fs/goose/concept/deps${PYTHONPATH:+:$PYTHONPATH}
export HF_MODULES_CACHE=/n/fs/goose/concept/hf_modules_test_extension
read -r CONCEPT_REVISION < /n/fs/goose/concept/model_revision.txt

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/concept/extract.py \
  --review "$EXT/manifest.csv" \
  --output "$EXT/detector" \
  --summary-dir "$EXT/detector" \
  --revision "$CONCEPT_REVISION" \
  --include-test

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/concept/category_counts.py \
  --predictions "$EXT/detector/concept_predictions.csv" \
  --records-dir "$EXT/detector" \
  --output "$EXT/annotations"

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/sae-training/evaluate_face_parts_extension.py \
  --data "$EXT" \
  --run /n/fs/goose/sae-training/runs/face_parts_sae_001

/n/fs/goose/el8403/conda-envs/reno/bin/python -u /n/fs/goose/sae-training/render_face_region_heatmaps.py \
  /n/fs/goose/sae-training/runs/face_parts_sae_001 \
  --data "$EXT" \
  --boxes "$EXT/annotations/bounding_boxes.csv" \
  --residuals "$EXT/residuals" \
  --output "$EXT/region_heatmaps"
