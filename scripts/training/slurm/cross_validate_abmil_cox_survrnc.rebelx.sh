#!/bin/bash
# cross_validate_abmil_cox_survrnc.rebelx.sh
#
# SLURM wrapper for scripts/training/cross_validate_abmil_cox_survrnc.py on the
# RebelX GPU cluster. Nested 5-fold CV over the whole TCGA-LUAD cohort on one A30:
#   - outer: patient-level, event-stratified 5-fold; every patient held out once
#   - inner: train_batch_size in {16, 32, 96}, selected on each fold's own val
#     patients, never on the held-out fold
#   - fixed baseline architecture (embed 128, attention 128); the 4-architecture
#     screen found no separation, so architecture is no longer swept
#
# 15 training runs (5 folds x 3 inner candidates). Per fold the split is roughly
# 310 train / 54 val / 92 test patients. Each epoch reads the uncapped val bags
# (~3.9 GB); the capped training pass is ~2.2 GB/epoch regardless of batch size.
# Expect ~0.6-1.2 TB of validation reads across the run, so wall clock is bound
# by feature I/O, not by the GPU.
#
# Outputs are written to:
#   runs/abmil_cox_survrnc_cv_uni_v2_TCGA_LUAD/
#     cv_fold_summary.csv             one row per outer fold, with the selection
#     cv_inner_selection.csv          every inner candidate, for auditing
#     cv_pooled_patient_predictions.csv
#     cv_summary.json
#
# Submit from the repository root:
#   sbatch scripts/training/slurm/cross_validate_abmil_cox_survrnc.rebelx.sh
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
# No --nodelist pin: let SLURM place the job on any free A30 in the partition.

set -euo pipefail

REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Running on $(hostname), SLURM set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# This partition does not bind a job to a specific GPU: SLURM leaves
# CUDA_VISIBLE_DEVICES=0 for every job, so jobs pile onto card 0 while the other
# cards sit idle (that is why we keep landing on a full GPU 0).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mapfile -t GPU_FREE < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
BEST_ID=-1
BEST_FREE=0
for id in "${!GPU_FREE[@]}"; do
    if [ "${GPU_FREE[$id]}" -gt "$BEST_FREE" ]; then
        BEST_FREE=${GPU_FREE[$id]}
        BEST_ID=$id
    fi
done
echo "Freest GPU on $(hostname): index ${BEST_ID}, ${BEST_FREE} MiB free"
if [ "$BEST_ID" -lt 0 ] || [ "$BEST_FREE" -lt 8000 ]; then
    echo "ERROR: no GPU on $(hostname) has >= 8000 MiB free. All cards busy; resubmit later."
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$BEST_ID
echo "Bound to GPU $CUDA_VISIBLE_DEVICES"

echo "Starting nested 5-fold CV (inner grid: lambda_rnc)"

srun --unbuffered python scripts/training/cross_validate_abmil_cox_survrnc.py
