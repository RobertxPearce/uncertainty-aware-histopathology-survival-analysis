#!/bin/bash
# tune_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/training/tune_abmil_cox_survrnc.py on the RebelX
# GPU cluster. Screens ABMIL Cox + SurvRNC architectures on one A30:
#   - 4 architectures: baseline, baseline_input_norm, deep_proj, wide
#     (230K to 1.18M parameters; embeddings 128 to 512)
#   - each repeated over 4 seeds and ranked on the mean, because one run's
#     patient C-index on the 92-patient validation split has a standard
#     deviation near 0.04
#   - dropout 0.25, SurvRNC λ 0.05, 1024 patches, batch size 96
#   - learning rate 5e-5, weight decay 1e-3
#
# 16 trials total. The test split is not evaluated. Outputs are written to:
#   runs/abmil_cox_survrnc_tuning_uni_v2_TCGA_LUAD/
#     tuning_trials.csv   one row per (architecture, seed)
#     tuning_summary.csv  aggregated per config, ranked by mean C-index
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

srun --unbuffered python scripts/training/tune_abmil_cox_survrnc.py
