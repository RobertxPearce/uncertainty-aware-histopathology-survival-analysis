#!/bin/bash
# mc_dropout_abmil_cox.rebelx.sh
#
# SLURM wrapper for scripts/training/mc_dropout_abmil_cox.py on the RebelX GPU
# cluster (one A30). MC-dropout uncertainty for the ABMIL + Cox survival model on
# TCGA-GBMLGG, in two phases:
#   Phase 1 - 5-fold CV: a fresh model per fold, MC-dropout on each held-out fold
#   Phase 2 - frozen-split refit: train on the CSV's train/val split, MC-dropout
#             on its held-out test set
# Reads the frozen split at
#   data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv
# and the pre-extracted UNI v2 bags it points to. Outputs (metric CSVs, per-slide
# predictions, learning-curve histories) land in
#   results/mc_dropout_abmil_cox_tcga_gbmlgg/.
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/mc_dropout_abmil_cox.rebelx.sh
#
#SBATCH --job-name=abmil_mcdropout
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/abmil_mcdropout_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting MC-dropout: 5-fold CV + frozen-split refit on TCGA-GBMLGG"

srun --unbuffered python scripts/training/mc_dropout_abmil_cox.py
