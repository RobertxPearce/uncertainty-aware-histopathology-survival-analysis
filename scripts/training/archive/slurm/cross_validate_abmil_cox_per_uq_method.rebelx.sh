#!/bin/bash
# cross_validate_abmil_cox_per_uq_method.rebelx.sh
#
# SLURM wrapper for scripts/training/cross_validate_abmil_cox_per_uq_method.py on
# the RebelX GPU cluster (one A30). 5-fold CV over TCGA-GBMLGG comparing
# MC-dropout / deep-ensemble / SNGP on the arch-screen winner (baseline_input_norm
# + pure Cox). Metric CSVs and figures are written to
# results/cv_abmil_cox_tcga_gbmlgg/.
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/cross_validate_abmil_cox_per_uq_method.rebelx.sh
#
#SBATCH --job-name=abmil_uq_cv
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_uq_cv_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting 5-fold CV of MC-dropout / deep-ensemble / SNGP"

srun --unbuffered python scripts/training/cross_validate_abmil_cox_per_uq_method.py
