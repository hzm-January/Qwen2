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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import torch

cuda = 'cuda:1'
class_num = 4


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-id", type=str, default="dir-id")
    parser.add_argument("--cls", type=str, default="multiple")
    parser.add_argument("--selected", type=int, default=0)
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


def main():
    logger.info('============================ sft multi-class test starts now =======================')
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

    predicts, scores = [], []
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

        response_token_ids = tokenizer(response.lower())['input_ids']
        response_logits = [torch.softmax(logit, dim=-1) for logit in output['logits']]

        label = label_info[i]

        predict = 0
        # if response == "forme fruste keratoconus":
        #     predict = 1
        # elif response == "subclinical keratoconus":
        #     predict = 2
        # elif response == "clinical keratoconus":
        #     predict = 3
        # elif response == "No":
        #     predict = 0
        # else:
        #     logger.warning(f'Error response : {response}')
        response_lower = response.lower()
        ff_token_ids = []
        if "forme fruste" in response_lower:
            predict = 1
            ff_token_ids = tokenizer('forme fruste')['input_ids']
        elif "subclinical" in response_lower:
            predict = 2
            ff_token_ids = tokenizer('subclinical')['input_ids']
        elif "clinical" in response_lower:
            predict = 3
            ff_token_ids = tokenizer('clinical')['input_ids']
        elif "no" in response_lower:
            predict = 0
            no_ids = [2753, 8996, 2152]
            ids_detail = [_id_ for _id_ in no_ids if
                          _id_ in response_token_ids]
            if ids_detail: ff_token_ids = [ids_detail[0]]
        else:
            logger.warning(f'Error response : {response_lower}')

        logits_classes = [0.0] * class_num
        if ff_token_ids:
            logits = 0.0
            ff_token_ids_len = len(ff_token_ids)
            start_i = -1
            for j in range(len(response_token_ids) - ff_token_ids_len + 1):
                if response_token_ids[j:j + ff_token_ids_len] == ff_token_ids:
                    start_i = j
                    break
            # ff_logits = response_logits[start_i: start_i+ff_token_ids_len][0][ff_token_ids]

            for i, token_id in enumerate(ff_token_ids):
                logits += response_logits[i + start_i][0][token_id].item()
            logits = logits / ff_token_ids_len
        else:
            logits = 1e-8

        logger.info(f"id: {i}, predict: {predict}, label: {label}, scores: {logits}")

        predicts.append(predict)
        logits_classes[predict] = logits
        scores.append(logits_classes)

        if label == predict: correct[label] += 1

        label_cnt[label] += 1

    path = Path(diagnose_test_dataset_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(diagnose_predict_result_json, 'w') as f:
        json.dump(predicts, f, ensure_ascii=False)

    logger.info(f"----    correct   ---- {correct}")
    logger.info(f"----  label count ---- {label_cnt}")
    accs = [corr / label_cnt[i] if label_cnt[i] != 0 else 0 for i, corr in enumerate(correct)]
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

    # Binarize the output (one-hot encoding)
    y_test_bin = label_binarize(predicts, classes=list(range(class_num)))
    # y_score_bin = label_binarize(scores, classes=list(range(class_num)))
    y_score_bin = np.array(scores)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(class_num):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    logger.info(f'roc_auc_micro: {roc_auc["micro"]}')

    # Macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= class_num
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    logger.info(f'roc_auc_macro: {roc_auc["macro"]}')

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle=':', linewidth=4,
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})')

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen']
    for i, color in enumerate(colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curve')
    plt.legend(loc="lower right")

    roc_path = os.path.join(args.path['dataset_dir'], 'sft_mc_roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    logger.info(f'roc path: {roc_path}')
    logger.info('============================ sft multi-class test finished =======================')

    # 显示图片
    plt.show()

if __name__ == '__main__':
    main()
