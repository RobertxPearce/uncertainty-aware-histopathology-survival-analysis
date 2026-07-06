#!/bin/bash
# extract_features.rebelx.sh
#
# SLURM wrapper for the RebelX GPU cluster. Paired with the shared, cluster-
# agnostic worker scripts/feature_extraction/extract_features_trident.py.
#
# Runs TRIDENT UNI feature extraction over every .svs under data/raw/slides on
# the A30 GPUs. Everything for the run lands under one dir,
# data/processed/features_uni_v1_full/:
#   features/ -> one UNI embedding .h5 per slide
#   geojson/  -> one tissue-segmentation .geojson per slide
#   work/     -> intermediate patch coords, thumbnails, contours
#
#SBATCH --job-name=trident_feat
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu002
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/trident_%A_%a.out

set -euo pipefail

# Paths
REPO_DIR=/home/sp00006/uncertainty-aware-histopathology-survival-analysis
SLIDES_DIR="$REPO_DIR/data/raw/slides"

cd "$REPO_DIR"
mkdir -p logs

# Environment
source /home/$USER/miniconda3/bin/activate
conda activate survivors

# The gpu nodes have NO internet, so model weights must already be cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Sharding (job arrays)
SHARD_IDX=${SLURM_ARRAY_TASK_ID:-0}
N_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
echo "Running shard ${SHARD_IDX}/${N_SHARDS} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Run
srun --unbuffered python scripts/feature_extraction/extract_features_trident.py \
    --slides-dir "$SLIDES_DIR" \
    --device cuda:0 \
    --num-workers 6 \
    --no-cleanup \
    --shard "${SHARD_IDX}/${N_SHARDS}"
