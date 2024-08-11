import os
import json
import yaml
import argparse
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--dir-id", type=str, default="20240725-104805")
    parser.add_argument('--config', type=str, default='/public/whr/hzm/code/qwen2/ai_docter/config/dataset_config.yaml')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f'Config file {args.config} does not exist')
    #
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)

    with open(args.config, 'r') as file_conf:
        file_args = yaml.safe_load(file_conf)

    # file_args.update(args.__dict__)

    # merge command configs and file configs
    for key, value in file_args.items():
        # add file config if it is not in command args.
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    return args


def main():
    args = load_config()
    # dir_id = '20240725-104805'
    dir_id = args.dir_id

    model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
    diagnose_test_dataset_json = model_dir + '/train_test_dataset/diagnose_test_dataset.json'
    diagnose_test_label_json = model_dir + '/train_test_dataset/diagnose_test_label.json'

    diagnose_test_dataset_dir = model_dir + '/predict_result/'
    diagnose_predict_result_json = diagnose_test_dataset_dir + 'diagnose_predict_result.json'

    F_T = 1  # positive flag
    F_F = 0  #

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,  # path to the output directory
        device_map="cuda:0",
        trust_remote_code=True,
        bf16=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not hasattr(tokenizer, 'model_dir'):
        tokenizer.model_dir = model_dir

    with open(diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = json.load(file)
    with open(diagnose_test_label_json, 'r') as file:
        label_info = json.load(file)

    patient_cnt = len(label_info)
    logger.info("---- data count ----: %d", patient_cnt)

    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    predicts = []
    for i in range(patient_cnt):
        # print(diagnose_test_dataset[i])
        query = diagnose_test_dataset[i] + ''
        response, history = model.chat(tokenizer,
                                       query=diagnose_test_dataset[i],
                                       history=None)
        logger.info(response)
        # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
        # response, history = model.chat(tokenizer, query=new_query, history=history)
        # label = F_T if label_info[i] else F_F
        # print(new_query)

        label = label_info[i]
        label = 1 if label else 0

        # predict = 1 if response == "Yes" else 0
        predict = 1 if 'yes' in response.lower() else 0
        logger.info("id: %d, predict: %d, label: %d", i, predict, label)

        predicts.append(predict)

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

    f1 = 2 * TP / (2 * TP + FP + FN)
    logger.info('TP: %d, FP: %d, TN: %d, FN: %d', TP, FP, TN, FN)
    logger.info('准确率：%f %f', correct / patient_cnt, (TP + TN) / (TP + FP + TN + FN))
    logger.info('灵敏度：%f', sensitivity)
    logger.info('特异度：%f', specificity)
    logger.info('F1-Score：%f', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test  checkpoint.")

    parser.add_argument(
        "-o", "--dir-id", type=str, default="20240725-104805"
    )

    args = parser.parse_args()

    main(args)
