import os
import json
import yaml
import argparse
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
cuda='cuda:4'

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-id", type=str, default="20240912-163458")
    parser.add_argument("--selected", type=int, default=0)
    parser.add_argument("--cls", type=str, default="single")
    parser.add_argument("--lora", type=int, default=0)
    parser.add_argument('--ds_config', type=str,
                        default='/data/whr/hzm/code/qwen2/ai_doctor/config/dataset_config.yaml')
    parser.add_argument('--ft_config', type=str,
                        default='/data/whr/hzm/code/qwen2/ai_doctor/config/finetune_config.yaml')
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


def eval():
    logger.info('============================ sft binary class test starts now =======================')
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
    base_model_dir = os.path.join(args.ft_path['base_model_dir'])
    model_dir = os.path.join(args.ft_path['sft_model_dir'], args.dir_id)
    diagnose_test_dataset_dir = os.path.join(model_dir, 'predict_result')

    diagnose_predict_result_json = os.path.join(diagnose_test_dataset_dir, 'diagnose_predict_result.json')

    F_T = 1  # positive flag
    F_F = 0  #

    if args.lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            device_map=cuda,
            torch_dtype="auto",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, model_dir)

        model.eval()
    else:
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

    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    predicts, predict_probs = [], []
    acc_predicts, acc_labels = [0, 0], [0, 0]
    for i in range(patient_cnt):
        # print(diagnose_test_dataset[i])
        prompt = args.prompt['finetune_diagnose_require']
        if args.cls.lower() == 'multiple': prompt = args.prompt['finetune_diagnose_require_mc']
        content = str(diagnose_test_dataset[i]) + '\n' + prompt
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

        output = model.generate(
            **model_inputs,
            max_new_tokens=100,
            return_dict_in_generate=True,
            # output_attentions=True,
            output_scores=True,
            output_logits=True,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output['sequences'])
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # if i < 5: logger.info(response)
        logger.info(response)
        # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
        # response, history = model.chat(tokenizer, query=new_query, history=history)
        # label = F_T if label_info[i] else F_F
        # print(new_query)

        response_token_ids = tokenizer(response)['input_ids']

        # No: 2753, NO: 8996, no: 2152,
        no_ids = [2753, 8996, 2152]
        ids_detail = [{'token_id_index': response_token_ids.index(_id_), 'token_id': _id_} for _id_ in no_ids if _id_ in response_token_ids]
        # logger.info(f'no_ids_detail: {ids_detail}')
        if not ids_detail:
            # Yes: 9454, YES: 14004,  yes: 9693,
            yes_ids = [9454, 14004, 9693]
            ids_detail = [{'token_id_index': response_token_ids.index(_id_), 'token_id': _id_} for _id_ in yes_ids if
                          _id_ in response_token_ids]
            # logger.info(f'yes_ids_detail: {ids_detail}')

        if not ids_detail:
            ids_detail = [{'token_id_index': response_token_ids.index(_id_), 'token_id': _id_} for _id_ in response_token_ids]
        response_logits = [torch.softmax(logit, dim=-1) for logit in output['logits']]
        token_id_index = ids_detail[0]['token_id_index']
        token_id = ids_detail[0]['token_id']
        token_logits = response_logits[token_id_index][0][token_id]
        if token_id in no_ids:
            token_logits = 1-token_logits

        predict_probs.append(token_logits.item())
        logger.info(f'token_id: {token_id}, token logits: {token_logits}')

        label = label_info[i]
        label = 1 if label else 0

        # predict = 1 if response == "Yes" else 0
        predict = 1 if 'yes' in response.lower() else 0
        # predict = 1 if 'forme fruste' in response.lower() else 0
        logger.info(f"id: {i}, predict: {predict}, label: {label}")

        predicts.append(predict)

        if predict == label:
            acc_predicts[predict] += 1
        acc_labels[label] += 1

        if label == predict: correct += 1

        if label == F_T and predict == F_T:
            TP += 1
        elif label == F_F and predict == F_T:
            FP += 1
        elif label == F_F and predict == F_F:
            TN += 1
        elif label == F_T and predict == F_F:
            FN += 1
        else:
            logger.info('Prediction is not Yes and not No either, It is %d, GT is %d', predict, label)

    path = Path(diagnose_test_dataset_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(diagnose_predict_result_json, 'w') as f:
        json.dump(predicts, f, ensure_ascii=False)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # f1_ = 2*precision*recall/(precision+recall)

    logger.info(f'acc_labels: {acc_labels}')
    logger.info(f'acc_predicts: {acc_predicts}')

    logger.info(f'0 acc: {acc_predicts[0] / acc_labels[0] if acc_labels[0] != 0 else 0}')
    logger.info(f'1 acc: {acc_predicts[1] / acc_labels[1] if acc_labels[1] != 0 else 0}')

    f1 = 2 * TP / (2 * TP + FP + FN)
    logger.info(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
    logger.info(f'准确率：{correct / patient_cnt} {(TP + TN) / (TP + FP + TN + FN)}')
    logger.info(f'灵敏度：{sensitivity}')
    logger.info(f'特异度：{specificity}')
    logger.info(f'F1-Score：{f1}')

    fpr, tpr, thresholds = roc_curve(label_info, predict_probs)
    logger.info(f'thresholds: {thresholds}')
    roc_auc = auc(fpr, tpr)
    logger.info(f'false positive rate ↓ : {fpr}')
    logger.info(f'true positive rate ↑ : {tpr}')
    logger.info(f'thresholds : {thresholds}')
    logger.info(f'roc_auc: {roc_auc}')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    roc_path = os.path.join(diagnose_test_dataset_dir, 'sft_bc_roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    # plt.show()
    logger.info(f'roc path: {roc_path}')
    logger.info('============================ sft binary class test finished =======================')
    return correct / patient_cnt

if __name__ == '__main__':
    eval()
