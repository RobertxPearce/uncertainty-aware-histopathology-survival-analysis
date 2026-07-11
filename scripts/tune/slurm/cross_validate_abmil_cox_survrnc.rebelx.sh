#!/bin/bash
# cross_validate_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/tune/cross_validate_abmil_cox_survrnc.py on the RebelX
# GPU cluster (one A30). Nested 5-fold CV over TCGA-LUAD (15 runs: 5 folds x 3
# inner candidates); outputs to runs/abmil_cox_survrnc_cv_uni_v2_TCGA_LUAD/.
#
# Submit from the repository root:
#   sbatch scripts/tune/slurm/cross_validate_abmil_cox_survrnc.rebelx.sh
#
#SBATCH --job-name=abmil_survrnc_cv
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_survrnc_cv_%j.out
#SBATCH --nodelist=gpu[002]

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), SLURM set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,memory.free --format=csv
echo "Starting nested 5-fold CV (inner grid: lambda_rnc)"

srun --unbuffered python scripts/tune/cross_validate_abmil_cox_survrnc.py
