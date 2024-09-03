#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CUDA_IDS=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8
CLS="single"

#DPO_CUDA_IDS=0,1,2,3,4,5,6
#CUDA_IDS=0,2,5,6,7
#CUDA_IDS=4,5,6,7

#export CUDA_VISIBLE_DEVICES=$CUDA_IDS
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

USE_LORA=False
Q_LORA=False
SELECTED=0
DIGIT_TO_WORD=1


MODEL="/public/whr/hzm/model/qwen2-base/qwen2/qwen2-7b-instruct"
# Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.

DS_CONFIG_PATH="/public/whr/hzm/code/qwen2/examples/sft/ds_config_zero3.json"

export PATH=/usr/local/cuda-12.1/bin:$PATH

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
#GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')


# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

MASTER_ADDR="172.18.127.64"
MASTER_PORT=6001
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

for i in {1..10}
do
  echo "Numver $i"
  DIR_ID=$(date '+%Y%m%d-%H%M%S')
  SFT_OUTPUT_DIR="/public/whr/hzm/model/qwen2-sft/$DIR_ID"
  DPO_OUTPUT_DIR="/public/whr/hzm/model/qwen2-dpo/$DIR_ID"

  SFT_TRAIN_DATA=$([ "$SELECTED" -eq 1 ] && echo "$SFT_OUTPUT_DIR/source/sft_fs_train_data.jsonl" || echo "$SFT_OUTPUT_DIR/source/sft_train_data.jsonl")
  SFT_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "/public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_sft_diagnose_test.py" || echo "/public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_sft_diagnose_test_multi_class.py")
  DPO_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "/public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_dpo_diagnose_test.py" || echo "/public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_dpo_diagnose_test_multi_class.py")


  echo "[DIGIT_TO_WORD]:  $DIGIT_TO_WORD, [SELECTED]: $SELECTED"

  #mkdir -p $SFT_OUTPUT_DIR
  #mkdir -p $DPO_OUTPUT_DIR

  echo "[SFT_OUTPUT_DIR]: $SFT_OUTPUT_DIR"
  echo "[SFT_TRAIN_DATA]: $SFT_TRAIN_DATA"


  DISTRIBUTED_ARGS="
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $NNODES \
      --node_rank $NODE_RANK \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
  "

  run_sh="CUDA_VISIBLE_DEVICES=$CUDA_IDS \
      /public/whr/anaconda3/envs/hzm-qwen2-01/bin/torchrun \
      $DISTRIBUTED_ARGS \
      /public/whr/hzm/code/qwen2/examples/sft/finetune.py \
      --model_name_or_path $MODEL \
      --data_path $SFT_TRAIN_DATA \
      --bf16 True \
      --output_dir $SFT_OUTPUT_DIR \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 1000 \
      --save_total_limit 10 \
      --learning_rate 1e-5 \
      --weight_decay 0.01 \
      --adam_beta2 0.95 \
      --warmup_ratio 0.01 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --report_to tensorboard \
      --model_max_length 8192 \
      --lazy_preprocess True \
      --use_lora ${USE_LORA} \
      --q_lora ${Q_LORA} \
      --gradient_checkpointing \
      --deepspeed ${DS_CONFIG_PATH} \
      "

  # 1 generate train and test dataset with selected features
  /public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 /public/whr/hzm/code/qwen2/ai_doctor/data/dataset_process.py --digit-to-word $DIGIT_TO_WORD --cls $CLS --selected $SELECTED 2>&1 | tee "$SFT_OUTPUT_DIR/train_model.log"
  # 2 copy dataset from ai_doctor to output_dir/source/
  mkdir -p $SFT_OUTPUT_DIR/source
  cp -r /public/whr/hzm/code/qwen2/ai_doctor/source/*.jsonl $SFT_OUTPUT_DIR/source

  mkdir -p $DPO_OUTPUT_DIR/source
  cp -r /public/whr/hzm/code/qwen2/ai_doctor/source/*.jsonl $DPO_OUTPUT_DIR/source

  # 3 sft train
  echo "[run_sh]: $run_sh"
  eval $run_sh 2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"
  ## 4 sft test
  /public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 $SFT_TEST_FILE_PATH --dir-id $DIR_ID --selected $SELECTED  2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"
  ## 5 dpo train
  deepspeed --include localhost:$CUDA_IDS /public/whr/hzm/code/qwen2/ai_doctor/dpo/main_train.py --dir_id $DIR_ID  --selected $SELECTED 2>&1 | tee "$DPO_OUTPUT_DIR/train_model.log"
  # 6 dpo test
  /public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 $DPO_TEST_FILE_PATH --dir-id $DIR_ID --selected $SELECTED 2>&1 | tee -a "$DPO_OUTPUT_DIR/train_model.log"
done