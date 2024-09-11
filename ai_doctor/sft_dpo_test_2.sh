#!/bin/bash

ROOT="/public/whr"
CODE_ROOT="$ROOT/hzm/code/qwen2"
MODEL_ROOT="$ROOT/hzm/model"
MODEL="$MODEL_ROOT/qwen2-base/qwen2/qwen2-7b-instruct"
# Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.

DS_CONFIG_PATH="$CODE_ROOT/examples/sft/ds_config_zero3.json"
for i in {1..1}
do
  DIR_ID=$(date '+%Y%m%d-%H%M%S')

  DIR_IDS_ARR[${#DIR_IDS_ARR[@]}]=$DIR_ID

done
# test
for CHECK_DIR_ID in "${DIR_IDS_ARR[@]}"
do
  echo ${CHECK_DIR_ID}
  SFT_OUTPUT_DIR="$MODEL_ROOT/qwen2-sft/$CHECK_DIR_ID"
  DPO_OUTPUT_DIR="$MODEL_ROOT/qwen2-dpo/$CHECK_DIR_ID"

  SFT_TRAIN_DATA=$([ "$SELECTED" -eq 1 ] && echo "$SFT_OUTPUT_DIR/source/sft_fs_train_data.jsonl" || echo "$SFT_OUTPUT_DIR/source/sft_train_data.jsonl")
  SFT_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_sft_diagnose_test_multi_class.py")
  DPO_TEST_FILE_PATH=$([ "$CLS" = "single" ] && echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test.py" || echo "$CODE_ROOT/ai_doctor/test/qwen2_dpo_diagnose_test_multi_class.py")

  ## 4 sft test
  /public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 $SFT_TEST_FILE_PATH --dir-id $CHECK_DIR_ID --selected $SELECTED 2>&1 | tee -a "$SFT_OUTPUT_DIR/train_model.log"
  # 6 dpo test
  /public/whr/anaconda3/envs/hzm-qwen2-01/bin/python3.9 $DPO_TEST_FILE_PATH --dir-id $CHECK_DIR_ID --selected $SELECTED 2>&1 | tee -a "$DPO_OUTPUT_DIR/train_model.log"
done