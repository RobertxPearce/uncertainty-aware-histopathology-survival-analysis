#!/bin/bash
# screen_architecture_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/tune/screen_architecture_abmil_cox_survrnc.py on the
# RebelX GPU cluster (one A30). K-fold CV architecture screen over TCGA-LUAD
# (4 architectures x 5 folds = 20 runs); outputs to
# runs/abmil_cox_survrnc_arch_screen_uni_v2_TCGA_LUAD/.
#
# Submit from the repository root:
#   sbatch scripts/tune/slurm/screen_architecture_abmil_cox_survrnc.rebelx.sh
#
#SBATCH --job-name=abmil_survrnc_arch
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_survrnc_arch_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting K-fold CV architecture screen (4 architectures x 5 folds)"

srun --unbuffered python scripts/tune/screen_architecture_abmil_cox_survrnc.py
