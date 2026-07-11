#!/bin/bash
# tune_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/tune/tune_abmil_cox_survrnc.py on the RebelX GPU
# cluster (one A30). Screens ABMIL Cox + SurvRNC architectures (4 architectures
# x 4 seeds, 16 trials); outputs to runs/abmil_cox_survrnc_tuning_uni_v2_TCGA_LUAD/.
#
# Submit from the repository root:
#   sbatch scripts/tune/slurm/tune_abmil_cox_survrnc.rebelx.sh
#
#SBATCH --job-name=abmil_survrnc_tune
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/abmil_survrnc_tune_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting ABMIL Cox + SurvRNC architecture screen (4 architectures x 4 seeds)"

srun --unbuffered python scripts/tune/tune_abmil_cox_survrnc.py
