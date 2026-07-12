#!/bin/bash
# run_uncertainty_full_abmil_cox.rebelx.sh
#
# SLURM wrapper for scripts/training/run_uncertainty_full_abmil_cox.py on the
# RebelX GPU cluster (one A30). Full uncertainty run over TCGA-GBMLGG: 5-fold CV
# selection across MC-dropout / deep-ensemble / SNGP, then a default-split refit
# of the winning method. Outputs to
# runs/abmil_cox_uncertainty_full_uni_v2_TCGA_GBMLGG/.
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/run_uncertainty_full_abmil_cox.rebelx.sh
#
#SBATCH --job-name=abmil_uq_full
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_uq_full_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting full uncertainty run (5-fold CV selection + default-split refit)"

srun --unbuffered python scripts/training/run_uncertainty_full_abmil_cox.py
