#!/bin/bash
# deep_ensembles_abmil_cox.rebelx.sh
#
# SLURM wrapper for scripts/training/deep_ensembles_abmil_cox.py on the RebelX GPU
# cluster (one A30). Deep-ensemble uncertainty for the ABMIL + Cox survival model
# on TCGA-GBMLGG, in two phases:
#   Phase 1 - 5-fold CV: an ensemble per fold, scored on each held-out fold
#   Phase 2 - frozen-split refit: an ensemble on the CSV's train/val split, scored
#             on its held-out test set (single-member baseline reported alongside)
# Reads the frozen split at
#   data/processed/experiments/uni_v2_TCGA_GBMLGG/splits.csv
# and the pre-extracted UNI v2 bags it points to. Outputs (metric CSVs, per-slide
# predictions, learning-curve histories, member checkpoints) land in
#   results/deep_ensembles_abmil_cox_tcga_gbmlgg/.
#
# This trains N_SPLITS*ENSEMBLE_SIZE + ENSEMBLE_SIZE (= 30) members, so it needs
# more walltime than the single-model UQ scripts.
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/deep_ensembles_abmil_cox.rebelx.sh
#
#SBATCH --job-name=abmil_ensemble
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_ensemble_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting deep ensembles: 5-fold CV + frozen-split refit on TCGA-GBMLGG"

srun --unbuffered python scripts/training/deep_ensembles_abmil_cox.py
