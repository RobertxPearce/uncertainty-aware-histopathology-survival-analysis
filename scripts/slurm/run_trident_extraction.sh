#!/bin/bash
#SBATCH --job-name=trident_feat      # 1. 작업 이름 지정 (원하는 이름으로)
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu001
#SBATCH --gres=gpu:1                 # 2. 필수 추가! GPU 1개를 할당해달라는 명령어입니다.
#SBATCH --output=logs/trident_%j.out # (선택) 로그가 터미널에 뜨지 않고 파일로 예쁘게 저장됩니다.

source /home/$USER/miniconda3/bin/activate
conda activate trident               # 3. 환경 이름 변경 (슈퍼컴의 가상환경 이름이 trident가 맞는지 확인!)

# 4. 파일의 정확한 경로 지정
srun --unbuffered python scripts/trident_feature_extraction.py