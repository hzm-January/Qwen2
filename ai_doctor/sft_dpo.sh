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
SELECTED=1
DIGIT_TO_WORD=0
#LABEL_SMOOTHING_FACTOR=0.1

ROOT="/data/whr"
CODE_ROOT="$ROOT/hzm/code/qwen2"
MODEL_ROOT="$ROOT/hzm/model"
MODEL="$MODEL_ROOT/qwen2-base/qwen2/qwen2-7b-instruct"
ALIGNED_MODEL="$MODEL_ROOT/qwen2-extra/qwen2/qwen2-7b-instruct-extra"
# Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.

DS_CONFIG_PATH="$CODE_ROOT/examples/sft/ds_config_zero3.json"

export PATH=/data/whr/cuda/cuda-12.1/bin:$PATH

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

MASTER_ADDR="172.18.30.23"
MASTER_PORT=6001
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

DIR_IDS_ARR=()

for i in {1..10}
do
  echo "Numver $i"
  DIR_ID=$(date '+%Y%m%d-%H%M%S')
  ALN_OUTPUT_DIR="$MODEL_ROOT/qwen2-aln/$DIR_ID"
  SFT_OUTPUT_DIR="$MODEL_ROOT/qwen2-sft/$DIR_ID"
  DPO_OUTPUT_DIR="$MODEL_ROOT/qwen2-dpo/$DIR_ID"

  ALN_TRAIN_DATA=$([ "$SELECTED" -eq 1 ] && echo "$SFT_OUTPUT_DIR/source/aln_fs_train_data.jsonl" || echo "$SFT_OUTPUT_DIR/source/aln_train_data.jsonl")
  SFT_TRAIN_DATA=$([ "$SELECTED" -eq 1 ] && echo "$SFT_OUTPUT_DIR/source/sft_fs_train_data.jsonl" || echo "$SFT_OUTPUT_DIR/source/sft_train_data.jsonl")
  SFT_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test_multi_class.py")
  DPO_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test_multi_class.py")


  echo "[DIGIT_TO_WORD]:  $DIGIT_TO_WORD, [SELECTED]: $SELECTED"

  #mkdir -p $SFT_OUTPUT_DIR
  #mkdir -p $DPO_OUTPUT_DIR

  echo "[ALN_OUTPUT_DIR]: $ALN_OUTPUT_DIR, [ALN_TRAIN_DATA]: $ALN_TRAIN_DATA"
  echo "[SFT_OUTPUT_DIR]: $SFT_OUTPUT_DIR, [SFT_TRAIN_DATA]: $SFT_TRAIN_DATA"
#  echo "[DPO_OUTPUT_DIR]: $DPO_OUTPUT_DIR, [DPO_TRAIN_DATA]: $DPO_TRAIN_DATA"


  DISTRIBUTED_ARGS="
      --nproc_per_node $GPUS_PER_NODE \
      --nnodes $NNODES \
      --node_rank $NODE_RANK \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
  "

  run_sh_aln="CUDA_VISIBLE_DEVICES=$CUDA_IDS \
      $ROOT/anaconda3/envs/hzm-qwen2/bin/torchrun \
      $DISTRIBUTED_ARGS \
      $CODE_ROOT/examples/sft/finetune.py \
      --model_name_or_path $ALIGNED_MODEL \
      --data_path $ALN_TRAIN_DATA \
      --bf16 True \
      --output_dir $ALN_OUTPUT_DIR \
      --fix_llm False\
      --fix_embed True\
      --fix_json_embed False\
      --num_train_epochs 5 \
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
#      --label_smoothing_factor ${LABEL_SMOOTHING_FACTOR}
run_sh_sft="CUDA_VISIBLE_DEVICES=$CUDA_IDS \
      $ROOT/anaconda3/envs/hzm-qwen2/bin/torchrun \
      $DISTRIBUTED_ARGS \
      $CODE_ROOT/examples/sft/finetune.py \
      --model_name_or_path $ALN_OUTPUT_DIR \
      --data_path $SFT_TRAIN_DATA \
      --bf16 True \
      --output_dir $SFT_OUTPUT_DIR \
      --fix_llm False \
      --fix_embed False \
      --fix_json_embed False \
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
#--label_smoothing_factor ${LABEL_SMOOTHING_FACTOR}

  # 1 generate train and test dataset with selected features
  /data/whr/anaconda3/envs/hzm-qwen2/bin/python3.9 $CODE_ROOT/ai_doctor/data/dataset_process.py --digit-to-word $DIGIT_TO_WORD --cls $CLS --selected $SELECTED 2>&1 | tee "$SFT_OUTPUT_DIR/train_model.log"

  # 2 copy dataset from ai_doctor to output_dir/source/
  mkdir -p $ALN_OUTPUT_DIR/source
  cp -r $CODE_ROOT/ai_doctor/source/*.jsonl $ALN_OUTPUT_DIR/source

  mkdir -p $SFT_OUTPUT_DIR/source
  cp -r $CODE_ROOT/ai_doctor/source/*.jsonl $SFT_OUTPUT_DIR/source

  mkdir -p $DPO_OUTPUT_DIR/source
  cp -r $CODE_ROOT/ai_doctor/source/*.jsonl $DPO_OUTPUT_DIR/source

  # 3 align train
  echo "[run_sh_aln]: $run_sh_aln"
  eval $run_sh_aln 2>&1 | tee -a "$ALN_OUTPUT_DIR/train_model.log"

  # 3 sft train
  echo "[run_sh_sft]: $run_sh_sft"
  eval $run_sh_sft 2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"

#  DIR_IDS_ARR[${#DIR_IDS_ARR[@]}]=$DIR_ID
  DIR_IDS_ARR+=($DIR_ID)

#  if [ "$FLAG" = USE_LORA ]; then
#    echo "use_lora"
#  fi

  ## 4 sft test
  /data/whr/anaconda3/envs/hzm-qwen2/bin/python3.9 $SFT_TEST_FILE_PATH --dir-id $DIR_ID --selected $SELECTED  2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"
  ## 5 dpo train
  deepspeed --include localhost:$CUDA_IDS $CODE_ROOT/ai_doctor/dpo/main_train.py --dir_id $DIR_ID  --selected $SELECTED 2>&1 | tee "$DPO_OUTPUT_DIR/train_model.log"
  # 6 dpo test
  /data/whr/anaconda3/envs/hzm-qwen2/bin/python3.9 $DPO_TEST_FILE_PATH --dir-id $DIR_ID --selected $SELECTED 2>&1 | tee -a "$DPO_OUTPUT_DIR/train_model.log"
done
echo "${DIR_IDS_ARR[@]}"
# test
#for CHECK_DIR_ID in "${DIR_IDS_ARR[@]}"
#do
#  echo "$CHECK_DIR_ID"
#  SFT_OUTPUT_DIR="$MODEL_ROOT/qwen2-sft/$CHECK_DIR_ID"
#  DPO_OUTPUT_DIR="$MODEL_ROOT/qwen2-dpo/$CHECK_DIR_ID"
#
#  SFT_TRAIN_DATA=$([ "$SELECTED" -eq 1 ] && echo "$SFT_OUTPUT_DIR/source/sft_fs_train_data.jsonl" || echo "$SFT_OUTPUT_DIR/source/sft_train_data.jsonl")
#  SFT_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test_multi_class.py")
#  DPO_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test_multi_class.py")
#
#  ## 4 sft test
#  /data/whr/anaconda3/envs/hzm-qwen2/bin/python3.9 $SFT_TEST_FILE_PATH --dir-id $CHECK_DIR_ID --selected $SELECTED 2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"
#  # 6 dpo test
#  /data/whr/anaconda3/envs/hzm-qwen2/bin/python3.9 $DPO_TEST_FILE_PATH --dir-id $CHECK_DIR_ID --selected $SELECTED 2>&1 | tee -a "$DPO_OUTPUT_DIR/train_model.log"
#done
#echo "${DIR_IDS_ARR[@]}"