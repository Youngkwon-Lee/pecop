#!/bin/bash
# PECoP 훈련 - HPC V100 환경용 스크립트

# 사용법: bash run_on_hpc.sh [task] [epochs] [batch_size]
# 예: bash run_on_hpc.sh Gait 8 8

TASK=${1:-Gait}
EPOCHS=${2:-8}
BS=${3:-8}

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_ROOT="${SCRIPT_DIR}/datasets/PD4T_frames"
VIDEO_ROOT="${HOME}/PD4T_Videos"  # 절대 경로 사용
PRETRAINED_MODEL="${SCRIPT_DIR}/pretrained_models/model_rgb.pth"

# 로그 디렉토리
mkdir -p logs

# 로그 파일
LOG_FILE="logs/training_${TASK}_$(date +%Y%m%d_%H%M%S).log"

echo "==============================================="
echo "PECoP Training on HPC V100"
echo "==============================================="
echo "Task: $TASK"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BS"
echo "GPU: NVIDIA V100 (16GB)"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG_FILE"
echo "==============================================="
echo ""

# 데이터 리스트 선택
if [ "$TASK" == "Gait" ]; then
    DATA_LIST="${SCRIPT_DIR}/data_lists/Gait_train.list"
elif [ "$TASK" == "Hand movement" ]; then
    DATA_LIST="${SCRIPT_DIR}/data_lists/Hand movement_train.list"
    TASK_DIR="Hand movement"
elif [ "$TASK" == "Finger tapping" ]; then
    DATA_LIST="${SCRIPT_DIR}/data_lists/Finger tapping_train.list"
    TASK_DIR="Finger tapping"
elif [ "$TASK" == "Leg agility" ]; then
    DATA_LIST="${SCRIPT_DIR}/data_lists/Leg agility_train.list"
    TASK_DIR="Leg agility"
else
    echo "Unknown task: $TASK"
    echo "Available tasks: Gait, Hand movement, Finger tapping, Leg agility"
    exit 1
fi

# 데이터 리스트 확인
if [ ! -f "$DATA_LIST" ]; then
    echo "Error: Data list not found: $DATA_LIST"
    exit 1
fi

echo "Data list: $DATA_LIST"
echo "Data root: $VIDEO_ROOT"
echo ""

# 훈련 실행
python "${SCRIPT_DIR}/train_simple.py" \
  --device cuda:0 \
  --epoch $EPOCHS \
  --bs $BS \
  --lr 0.001 \
  --data_list "$DATA_LIST" \
  --video_root "$VIDEO_ROOT" \
  --pretrained_i3d_weight "$PRETRAINED_MODEL" \
  --dataset PD4T \
  --model i3d \
  --max_sr 5 \
  --max_segment 4 \
  --clip_len 32 \
  --crop_sz 224 \
  2>&1 | tee "$LOG_FILE"

# 훈련 완료
echo ""
echo "==============================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "==============================================="
