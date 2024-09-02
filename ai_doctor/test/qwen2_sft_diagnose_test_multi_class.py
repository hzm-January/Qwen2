import os
import json
import yaml
import argparse
from pathlib import Path
from loguru import logger
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

cuda = 'cuda:1'
class_num = 4


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-id", type=str, default="dir-id")
    parser.add_argument("--cls", type=str, default="multiple")
    parser.add_argument("--selected", type=int, default=0)
    parser.add_argument('--ds_config', type=str,
                        default='/public/whr/hzm/code/qwen2/ai_doctor/config/dataset_config.yaml')
    parser.add_argument('--ft_config', type=str,
                        default='/public/whr/hzm/code/qwen2/ai_doctor/config/finetune_config.yaml')
    args = parser.parse_args()

    if not os.path.exists(args.ds_config):
        logger.error(f'Config file {args.ds_config} does not exist')
    #
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)

    print(f'Loading config from {args.ds_config}')
    with open(args.ds_config, 'r') as file_conf:
        file_args = yaml.safe_load(file_conf)

    # file_args.update(args.__dict__)

    # merge command configs and file configs
    for key, value in file_args.items():
        # add file config if it is not in command args.
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    with open(args.ft_config, 'r') as file_conf:
        file_ft_args = yaml.safe_load(file_conf)

    for key, value in file_ft_args.items():
        # add file config if it is not in command args.
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def main():
    args = load_config()
    # dir_id = '20240725-104805'
    # dir_id = '20240811-131954'
    # dir_id = '20240811-134819' # '20240811-230536'
    # dir_id = '20240812-105343' # word + all feature

    if args.selected:
        diagnose_test_dataset_json = os.path.join(args.path['dataset_dir'], args.file_name['test_fs_data'])
        diagnose_test_label_json = os.path.join(args.path['dataset_dir'], args.file_name['test_fs_label'])
    else:
        diagnose_test_dataset_json = os.path.join(args.path['dataset_dir'], args.file_name['test_data'])
        diagnose_test_label_json = os.path.join(args.path['dataset_dir'], args.file_name['test_label'])

    # dir_id = '20240811-230536'  #
    #
    # diagnose_test_dataset_json = os.path.join(args.path['dataset_dir'], args.file_name['test_fs_data'])
    # diagnose_test_label_json = os.path.join(args.path['dataset_dir'], args.file_name['test_fs_label'])
    model_dir = os.path.join(args.ft_path['sft_model_dir'], args.dir_id)
    diagnose_test_dataset_dir = os.path.join(model_dir, 'predict_result')

    diagnose_predict_result_json = os.path.join(diagnose_test_dataset_dir, 'diagnose_predict_result.json')

    F_T = 1  # positive flag
    F_F = 0  #

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,  # path to the output directory
        torch_dtype="auto",
        device_map=cuda,
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.ft_path['base_model_dir'], trust_remote_code=True)
    if not hasattr(tokenizer, 'base_model_dir'):
        tokenizer.model_dir = args.ft_path['base_model_dir']

    with open(diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = [json.loads(line) for line in file]
    with open(diagnose_test_label_json, 'r') as file:
        label_info = [json.loads(line) for line in file]

    patient_cnt = len(label_info)
    logger.info(f"---- data count ----: {patient_cnt}")

    correct = [0] * class_num
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    predicts = []
    label_cnt = [0] * class_num
    for i in range(patient_cnt):
        # print(diagnose_test_dataset[i])
        prompt = args.prompt['finetune_diagnose_require']
        if args.cls == 'multiple': prompt = args.prompt['finetune_diagnose_require_mc']
        content = diagnose_test_dataset[i] + '\n' + prompt
        messages = [
            {"role": "system", "content": "You are an ophthalmology specialist."},
            {"role": "user", "content": content}
        ]
        if i == 0: logger.info(f'=============== sft test message: {messages}')
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(cuda)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if i < 5: logger.info(response)
        # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
        # response, history = model.chat(tokenizer, query=new_query, history=history)
        # label = F_T if label_info[i] else F_F
        # print(new_query)

        label = label_info[i]

        predict = 0
        if response == "forme fruste keratoconus":
            predict = 1
        elif response == "subclinical keratoconus":
            predict = 2
        elif response == "clinical keratoconus":
            predict = 3
        elif response == "No":
            predict = 0
        else:
            logger.warning(f'Error response : {response}')

        logger.info(f"id: {i}, predict: {predict}, label: {label}")

        predicts.append(predict)

        if label == predict: correct[label] += 1

        label_cnt[label] += 1

    path = Path(diagnose_test_dataset_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(diagnose_predict_result_json, 'w') as f:
        json.dump(predicts, f, ensure_ascii=False)

    logger.info(f"----    correct   ---- {correct}")
    logger.info(f"----  label count ---- {label_cnt}")
    accs = [corr / label_cnt[i] for i, corr in enumerate(correct)]
    logger.info(f"---- accuracy ---- {accs}")

    # 准确率
    accuracy = accuracy_score(label_info, predicts)
    logger.info(f"Accuracy: {accuracy}")
    # 混淆矩阵
    conf_matrix = confusion_matrix(label_info, predicts)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    # 分类报告
    class_report = classification_report(label_info, predicts)
    logger.info(f"Classification Report:\n{class_report}")
    # 精确率
    precision = precision_score(label_info, predicts, average='macro')
    logger.info(f"Precision: {precision}")
    # 召回率
    recall = recall_score(label_info, predicts, average='macro')
    logger.info(f"Recall: {recall}")
    # F1 分数
    f1 = f1_score(label_info, predicts, average='macro')
    logger.info(f"F1 Score: {f1}")

    # Specificity for each class
    specificity = []
    for i in range(len(conf_matrix)):
        tn = np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - np.diag(conf_matrix)[i])
        fp = np.sum(conf_matrix[:, i]) - np.diag(conf_matrix)[i]
        specificity.append(tn / (tn + fp))
    specificity = np.array(specificity)
    logger.info(f"specificity: {specificity}")


if __name__ == '__main__':
    main()
