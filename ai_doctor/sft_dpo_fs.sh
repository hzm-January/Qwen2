#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

#CUDA_IDS=0,1,2,3,4,5,6,7

#export CUDA_VISIBLE_DEVICES=$CUDA_IDS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DIR_ID=$(date '+%Y%m%d-%H%M%S')
OUTPUT_DIR="/public/whr/hzm/model/qwen2-sft/$DIR_ID"

MODEL="/public/whr/hzm/model/qwen2-base/qwen2/qwen2-7b-instruct"
# Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.
DATA="$OUTPUT_DIR/source/sft_fs_train_data.jsonl"
DS_CONFIG_PATH="/public/whr/hzm/code/qwen2/examples/sft/ds_config_zero3.json"
USE_LORA=False
Q_LORA=False

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')

# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

run_sh="/public/whr/anaconda3/envs/hzm-qwen2-01/bin/torchrun $DISTRIBUTED_ARGS /public/whr/hzm/code/qwen2/examples/sft/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
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
    --deepspeed ${DS_CONFIG_PATH}
    "

mkdir -p $OUTPUT_DIR

echo $OUTPUT_DIR

# 1 generate train and test dataset with selected features
/public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 /public/whr/hzm/code/qwen2/ai_doctor/data/dataset_process_feature_select.py 2>&1 | tee "$OUTPUT_DIR/train_model.log"
# 2 copy dataset from ai_doctor to output_dir/source/
mkdir -p $OUTPUT_DIR/source
cp -r /public/whr/hzm/code/qwen2/ai_doctor/source/*.jsonl $OUTPUT_DIR/source
# 3 sft train
eval $run_sh 2>&1 | tee "$OUTPUT_DIR/train_model.log"
## 4 sft test
/public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 /public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_sft_diagnose_test.py --dir-id $DIR_ID  2>&1 | tee "$OUTPUT_DIR/train_model.log"
## 5 dpo train
deepspeed --include localhost:$CUDA_IDS main_train.py --dir-id $DIR_ID
# 6 dpo test
/public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 /public/whr/hzm/code/qwen2/ai_doctor/test/qwen2_dpo_diagnose_test.py --dir-id $DIR_ID