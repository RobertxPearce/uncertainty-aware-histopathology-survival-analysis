#!/bin/bash
# run_trident_extraction.sh
#
# SLURM batch script to run the full TRIDENT feature-extraction pipeline on the
# RebelX GPU cluster.

#SBATCH --job-name=trident_feat
#SBATCH --partition=gpuq-a30           # education account -> A30 nodes (gpu001/gpu002)
#SBATCH --nodelist=gpu001              # pick a node; check `ssh gpu001 nvidia-smi` first
#SBATCH --gres=gpu:4                   # GPUs to request (node has 8x A30); raise up to 8
#SBATCH --cpus-per-task=16             # dataloader workers; ~4 CPUs per GPU
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trident_feat_%j.out   # %j = SLURM job id
#SBATCH --error=logs/trident_feat_%j.out

set -euo pipefail

# Walk up from SLURM_SUBMIT_DIR to the project root ($0 points at a spool copy
# under SLURM, not the repo).
MARKER="scripts/full_trident_feature_extraction.py"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -e "$PROJECT_ROOT/$MARKER" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -e "$PROJECT_ROOT/$MARKER" ]; then
    echo "ERROR: could not locate project root (looking for $MARKER) starting from ${SLURM_SUBMIT_DIR:-$PWD}" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
mkdir -p logs

# --- Environment (RebelX `survivors` conda env) ---
source /home/"$USER"/miniconda3/bin/activate
conda activate survivors

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv || true

# --- Run extraction ---
# Swap the paths below for the full dataset.
srun --unbuffered python scripts/full_trident_feature_extraction.py \
    --survival-table data/interim/matched_clinical_pilot100.csv \
    --slides-dir     data/raw/slides/pilot100_slides \
    --job-dir        data/processed/trident_full \
    --gpus auto

echo "Done."
