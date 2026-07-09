#!/bin/bash
# evaluate_abmil_model.rebelx.sh
#
# Evaluate the configured ABMIL checkpoint on the RebelX GPU cluster.
# Model, split, batch size, and output paths are defined at the top of:
#   scripts/eval/evaluate_abmil_model.py
#
# Submit from the repository root:
#   sbatch scripts/eval/slurm/evaluate_abmil_model.rebelx.sh
#
#SBATCH --job-name=abmil_eval
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/abmil_eval_%j.out
#SBATCH --nodelist=gpu002

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting ABMIL model evaluation"

srun --unbuffered python scripts/eval/evaluate_abmil_model.py
