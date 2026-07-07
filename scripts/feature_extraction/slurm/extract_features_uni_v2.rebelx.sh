#!/bin/bash
# extract_features_uni_v2.rebelx.sh
#
# SLURM wrapper for the RebelX GPU cluster. Paired with the cluster-agnostic
# worker scripts/feature_extraction/extract_features_trident_uni_v2.py.
#
# Runs TRIDENT UNI v2 (UNI2-h) feature extraction over every .svs under
# data/raw/slides on the A30 GPUs. Outputs follow the repo layout (cohort
# defaults to TCGA_LUAD):
#   data/processed/features/uni_v2/<cohort>/ -> one UNI v2 embedding .h5 per slide
#   data/processed/trident/<cohort>/geojson/ -> one tissue-seg .geojson per slide
#   data/processed/trident/<cohort>/work/    -> intermediate patch coords, contours
#
#SBATCH --job-name=trident_feat_v2
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trident_v2_%A_%a.out

set -euo pipefail

# Paths
REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/$USER/miniconda3/bin/activate
conda activate survivors

# The gpu nodes have NO internet, so model weights must already be cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Job-array sharding, device, workers, and paths are all configured as constants
# inside the Python script; the shard is read from SLURM_ARRAY_TASK_ID/COUNT.
echo "Running array task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Run
srun --unbuffered python scripts/feature_extraction/extract_features_trident_uni_v2.py
