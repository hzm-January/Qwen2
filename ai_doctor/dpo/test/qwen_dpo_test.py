import torch, os, json
import argparse
# from modelscope import (
#     snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# )
from pathlib import Path
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    # dir_id = '20240725-104805' 20240731-100541
    dir_id = args.dir_id

    token_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/'
    # base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/' + dir_id
    base_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/input-model/v2'
    dpo_model_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/output-model/result_10/checkpoint-1430'
    diagnose_test_dataset_json = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/dpo/dpo_test_dataset.json'
    diagnose_test_label_json = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/dpo/dpo_test_label.json'

    diagnose_test_dataset_dir = dpo_model_dir + '/predict_result/'
    diagnose_predict_result_json = diagnose_test_dataset_dir + 'dpo_predict_result.json'

    F_T = 1  # positive flag
    F_F = 0  #



    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_dir,  # path to the output directory
    #     device_map="cuda:0",
    #     trust_remote_code=True,
    #     bf16=True
    # )
    #
    # model = PeftModel.from_pretrained(base_model, dpo_model_dir, device_map="cuda:0", trust_remote_code=True,
    #                                   bf16=True)

    # tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="cuda:0",
        torch_dtype="auto",
        trust_remote_code=True,
        bf16=True
    )

    model = PeftModel.from_pretrained(base_model, dpo_model_dir)

    model.eval()
    # merged_model = model.merge_and_unload()
    # max_shard_size and safe serialization are not necessary.
    # They respectively work for sharding checkpoint and save the model to safetensors

    tokenizer = AutoTokenizer.from_pretrained(token_dir, trust_remote_code=True)
    if not hasattr(tokenizer, 'model_dir'):
        tokenizer.model_dir = base_model_dir

    with open(diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = json.load(file)
    with open(diagnose_test_label_json, 'r') as file:
        label_info = json.load(file)

    patient_cnt = len(label_info)
    print("---- data count ----: ", patient_cnt)

    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    predicts = []
    for i in range(patient_cnt):
        # print(diagnose_test_dataset[i])
        query = (diagnose_test_dataset[i]
                 +'Please answer the question with yes or no.'
                 # +'You must answer the question with yes or no, and the probability of your answer.'
                 #        ' Example output: yes or no,  float probability.'
                 )
        response, history = model.chat(tokenizer,
                                       query=query,
                                       history=None)
        print(response)
        # new_query = diagnose_test_dataset[i]+"请根据检测结果诊断该人员是否患有圆锥角膜病。"
        # response, history = model.chat(tokenizer, query=new_query, history=history)
        # label = F_T if label_info[i] else F_F
        # print(new_query)

        label = label_info[i]
        label = 1 if label else 0

        predict = 1 if "yes" in response.lower() else 0
        print("id:", i, "predict: ", predict, "label: ", label)

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
            print('Prediction is not Yes and not No either, It is ', predict, ', GT is ', label)

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

    print('TP: ', TP, ' FP: ', FP, ' TN: ', TN, ' FN: ', FN)
    print('准确率：', correct / patient_cnt, (TP + TN) / (TP + FP + TN + FN))
    print('灵敏度：', sensitivity)
    print('特异度：', specificity)
    print('F1-Score：', f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test  checkpoint.")

    parser.add_argument(
        "-o", "--dir-id", type=str, default="20240801-184826"
    )

    args = parser.parse_args()

    main(args)
