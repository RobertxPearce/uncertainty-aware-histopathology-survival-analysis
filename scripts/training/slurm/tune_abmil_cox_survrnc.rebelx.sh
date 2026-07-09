#!/bin/bash
# tune_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/training/tune_abmil_cox_survrnc.py on the RebelX
# GPU cluster. Runs the focused ABMIL Cox + SurvRNC validation grid on one A30:
#   - dropout:     0.25, 0.40
#   - SurvRNC λ:  0, 0.01, 0.05, 0.10
#   - 1024 patches, batch size 96, 128-dimensional embedding
#   - learning rate 5e-5, weight decay 1e-3
#
# The test split is not evaluated. Outputs are written to:
#   runs/abmil_cox_survrnc_tuning_uni_v2_TCGA_LUAD/
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/tune_abmil_cox_survrnc.rebelx.sh
#
#SBATCH --job-name=abmil_survrnc_tune
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
echo "Starting focused ABMIL Cox + SurvRNC tuning"

srun --unbuffered python scripts/training/tune_abmil_cox_survrnc.py \
    --device cuda \
    --num-workers 4
