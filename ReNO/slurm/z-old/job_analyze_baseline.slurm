#!/bin/bash
#SBATCH --job-name=analyze_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=el8403@princeton.edu

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate /n/fs/goose/el8403/conda-envs/reno

cd /n/fs/goose/ReNO

DATA_DIR=/n/fs/goose/z-old-data/racial_composition_baseline
WEIGHTS=fairface_weights/res34_fair_align_multi_7_20190809.pt

python scripts/racial_composition_analyze.py \
  --data_dir "$DATA_DIR" \
  --fairface_weights "$WEIGHTS" \
  --out_csv "$DATA_DIR/composition.csv" \
  --out_plot_dir "$DATA_DIR/plots"

python scripts/racial_composition_report.py \
  --csv "$DATA_DIR/composition.csv" \
  --out_md "$DATA_DIR/report.md" \
  --out_plot "$DATA_DIR/plots/final_composition.png"

python scripts/racial_composition_grid.py \
  --data_dir "$DATA_DIR" \
  --n_seeds_to_show 5 \
  --iterations 1 20 40 50 \
  --out_dir "$DATA_DIR/plots"
