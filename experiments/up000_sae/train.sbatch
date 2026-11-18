#!/bin/bash
#SBATCH --job-name=up000-sae-fit
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=12:00:00
#SBATCH --output=/n/fs/goose/experiments/up000_sae/fit-%j.out
set -euo pipefail
cd /n/fs/goose
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/n/fs/goose/layers/mid0_stability/generation_deps
py=el8403/conda-envs/goose/bin/python
run=experiments/up000_sae
source_data=sae-training/runs/face_parts_sae_001/test_extension
block=unet.up_blocks.0.attentions.0
checkpoint=sdxl-unbox/checkpoints/${block}_k10_hidden5120_auxk256_bs4096_lr0.0001/final
exec 9>"$run/fit.lock"
flock -n 9
if [[ ! -f "$run/capture/settings.json" ]]; then
    "$py" -u "$run/capture.py"
fi
fit="$run/training"
if [[ ! -f "$fit/settings.json" ]]; then
    if [[ -e "$fit" ]]; then
        mv "$fit" "$run/incomplete-training-$SLURM_JOB_ID"
    fi
    "$py" -u sae-training/fine_tune_face_sae.py \
        --residuals "$run/capture/residuals" --manifest "$run/capture/manifest.csv" \
        --boxes "$source_data/annotations/bounding_boxes.csv" --image-root "$source_data" \
        --checkpoint "$checkpoint" --block "$block" --output "$fit" \
        --epochs 20 --snapshot-interval 4 --lr 1e-6 --seed 20261101
fi
sha256sum "$fit/sae/"* > "$fit/checkpoint.sha256"
