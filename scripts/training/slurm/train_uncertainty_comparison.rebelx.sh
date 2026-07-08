#!/bin/bash
# train_uncertainty_comparison.rebelx.sh
#
# SLURM wrapper for scripts/training/train_abmil_cox_survrnc_uncertanty_comparison.py
# on the RebelX GPU cluster.
#
# Trains the ABMIL + Cox (+ SurvRNC) survival model three ways and compares their
# uncertainty on the TCGA-LUAD UNI v2 test split, on one A30 GPU:
#   - Deep Ensemble : ENSEMBLE_SIZE independently trained members
#   - MC Dropout    : stochastic passes through ensemble member 0
#   - SNGP          : one spectral-normalised model + fitted GP covariance
# Reads the frozen split at
#   data/processed/experiments/uni_v2_luad/splits.csv
# and the pre-extracted bags under data/processed/features/uni_v2/TCGA_LUAD/.
# Outputs:
#   runs/<EXPERIMENT_NAME>/            -> checkpoints, per-method prediction CSVs,
#                                         uncertainty_comparison.csv
#   results/figures/<EXPERIMENT_NAME>/ -> the comparison + diagnostic PDFs
#
#SBATCH --job-name=abmil_uq_compare
#SBATCH --partition=gpuq-a30
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/uq_comparison_%j.out
## Optional: pin to a specific node (leave commented to let SLURM pick within the partition)
## SBATCH --nodelist=gpu002

set -euo pipefail

# Paths
REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/$USER/miniconda3/bin/activate
conda activate survivors

# Headless plotting
export MPLBACKEND=Agg
export MPLCONFIGDIR="$REPO_DIR/logs/.mplcache"
mkdir -p "$MPLCONFIGDIR"

# All run settings are in script
echo "Running on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Run
srun --unbuffered python scripts/training/train_abmil_cox_survrnc_uncertanty_comparison.py
