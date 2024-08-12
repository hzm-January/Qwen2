import argparse
import json
import re
import os
import yaml
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel
from loguru import logger
from transformers.generation import GenerationConfig

column_name_json = '/public/whr/hzm/code/qwen2/ai_docter/source/patient_infos_column_name_1_2.json'

generation_config_dir = '/public/whr/hzm/code/qwen2/ai_docter/feature_select/'

cuda = "cuda:1"


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--dir-id", type=str, default="20240725-104805")
    parser.add_argument('--ds_config', type=str,
                        default='/public/whr/hzm/code/qwen2/ai_docter/config/dataset_config.yaml')
    parser.add_argument('--ft_config', type=str,
                        default='/public/whr/hzm/code/qwen2/ai_docter/config/finetune_config.yaml')
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
    sft_dir_id = '20240811-134819'
    dpo_dir_id = '20240811-210134'  # '20240811-201031'

    diagnose_test_dataset_json = os.path.join(args.path['dataset_dir'], args.file_name['test_data'])
    diagnose_test_label_json = os.path.join(args.path['dataset_dir'], args.file_name['test_label'])
    base_model_dir = os.path.join(args.ft_path['sft_model_dir'], sft_dir_id)
    dpo_model_dir = os.path.join(args.ft_path['dpo_model_dir'], dpo_dir_id)

    # 1 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.ft_path['base_model_dir'], trust_remote_code=True)

    generation_config = GenerationConfig.from_pretrained(generation_config_dir)
    logger.info(f'generation config: {generation_config}')

    # 2 model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map=cuda,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, dpo_model_dir)
    model.eval()

    # 3 test dataset
    with open(diagnose_test_dataset_json, 'r') as file:
        diagnose_test_dataset = [json.loads(line) for line in file]
    with open(diagnose_test_label_json, 'r') as file:
        label_info = [json.loads(line) for line in file]

    # 获取attention
    with open(column_name_json, 'r') as f:
        column_names = json.load(f)

    # 4
    total_len = len(label_info)
    for i in range(total_len):
        content = diagnose_test_dataset[i] + '\n' + args.prompt['finetune_diagnose_require']
        messages = [
            {"role": "system", "content": "You are an ophthalmology specialist."},
            {"role": "user", "content": content}
        ]
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
            output_attentions=True,
            output_scores=True,
        )

        logger.info(f'-------- content: {content}')
        logger.info(f'-------- content length: {len(content)}')
        logger.info(f'-------- model_inputs: {model_inputs}')
        logger.info(f'-------- input_ids: {model_inputs["input_ids"]}')
        logger.info(f'-------- input_ids shape: {model_inputs["input_ids"].shape}')
        # logger.info(f'-------- token_type_ids: {model_inputs["token_type_ids"]}')
        # logger.info(f'-------- token_type_ids shape: {model_inputs["token_type_ids"].shape}')
        logger.info(f'-------- attention_mask: {model_inputs["attention_mask"]}')
        logger.info(f'-------- attention_mask shape: {model_inputs["attention_mask"].shape}')
        # logger.info(f'-------- offset mapping: {model_inputs["offsets_mapping"]}')

        input_ids_tensor = model_inputs["input_ids"]
        input_ids_len = input_ids_tensor.shape[1]
        input_ids = input_ids_tensor[0].tolist()

        logger.info(f'input_ids: {input_ids}')

        # stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]

        # output = model.generate(
        #     **model_inputs,
        #     # stop_words_ids=stop_words_ids,
        #     generation_config=generation_config,
        #     # max_new_tokens=100,
        #     return_dict_in_generate=True,
        #     output_attentions=True,
        #     output_scores=True,
        # )
        logger.info(f'-------- output type: {type(output)}')
        logger.info(f'-------- output sequences type: {type(output["sequences"])}')
        logger.info(f'-------- output sequences shape: {output["sequences"].shape}')
        logger.info(f'-------- output sequences[0] shape: {output["sequences"][0].shape}')
        logger.info(f'-------- output sequences[0]: {output["sequences"][0]}')
        attentions = output["attentions"]
        # logger.info(f'-------- output attention: {attentions}')
        # logger.info(f'-------- output attention length: {len(attentions)}')
        # logger.info(f'-------- output attention [-1]: {attentions[-1]}')
        logger.info(f'-------- output attention [-1] length: {len(attentions[-1])}')  # 取输出的最后一个token对应的attentions
        logger.info(f'-------- output attention [-1][-1]: {attentions[-1][-1]}')  # 取输出的最后一个token对应的最后一层attention
        logger.info(
            f'-------- output attention [-1][-1] shape: {attentions[-1][-1].shape}')  # 取输出的最后一个token对应的最后一层attention

        attention = torch.mean(attentions[-1][-1], axis=1)[0].float()
        logger.info(f'-------- attention: {attention}')
        logger.info(f'-------- attention shape: {attention.shape}')

        output_token_decode_str = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)
        logger.info(f'-------- output_str: {output_token_decode_str}')
        logger.info(f'-------- output_str len: {len(output_token_decode_str)}')

        answer = output_token_decode_str[len(content) - 1:]
        logger.info(f'-------- answer: {answer}')

        attn_map = {}

        for i, column_name in enumerate(column_names):
            if 'Stress' in column_name: continue  # TODO: 待修复，Stress-strain 匹配不到
            cur_strs = re.findall(fr'({re.escape(column_name)}.*?),', content)
            logger.info(f'{i}----{column_name}-----{cur_strs}')
            cur_str = cur_strs[0]
            cur_str_tokens = tokenizer(cur_str, return_tensors='pt').to(cuda)
            cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
            # logger.info(f'----cur_str_ids: {cur_str_ids}')
            cur_str_ids_len = len(cur_str_ids)
            start_i = -1
            for j in range(input_ids_len - cur_str_ids_len + 1):
                if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                    start_i = j
                    break

            # TODO: 待解决，B,H等开头的英文字母会和前面的逗号一起作为token
            if start_i == -1:
                cur_str_tokens = tokenizer(',' + cur_str, return_tensors='pt').to(cuda)
                cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
                # logger.info(f'----cur_str_ids: {cur_str_ids}')
                cur_str_ids_len = len(cur_str_ids)
                for j in range(input_ids_len - cur_str_ids_len + 1):
                    if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                        start_i = j
                        break
            # TODO: 待解决，B,H等开头的英文字母会和前面的逗号一起作为token
            if start_i == -1:
                cur_str_tokens = tokenizer(' ' + cur_str, return_tensors='pt').to(cuda)
                cur_str_ids = cur_str_tokens["input_ids"][0].tolist()
                # logger.info(f'----cur_str_ids: {cur_str_ids}')
                cur_str_ids_len = len(cur_str_ids)
                for j in range(input_ids_len - cur_str_ids_len + 1):
                    if input_ids[j:j + cur_str_ids_len] == cur_str_ids:
                        start_i = j
                        break
            print(start_i)
            attn = attention[-1][start_i: start_i + cur_str_ids_len]
            logger.info(f'-------- attn: {attn}')
            attn_sum = torch.sum(attn).item()
            attn_map[column_name] = attn_sum

        sorted_items = sorted(attn_map.items(), key=lambda item: item[1], reverse=False)
        sorted_attn_map = {key: value for key, value in sorted_items}
        logger.info(f'-------- attention: {sorted_attn_map}')

        # keys, values = zip(*sorted_items[:30])
        keys, values = zip(*sorted_items)
        # 绘制条形图
        plt.figure(figsize=(20, 50))
        bars = plt.barh(keys, values, color='blue')

        # 在条形图上显示数值
        for bar in bars:
            plt.text(
                bar.get_width(),  # X 坐标（条形的宽度，即数值）
                bar.get_y() + bar.get_height() / 2,  # Y 坐标（条形的中点）
                f'{bar.get_width()}',  # 显示的文本（数值）
                va='center',  # 垂直对齐方式
                ha='left',  # 水平对齐方式
                color='black',  # 文本颜色
                fontsize=8  # 文本字体大小
            )
        plt.xlabel('Values', fontsize=8, fontweight='bold', color='black')
        plt.ylabel('Items', fontsize=8, fontweight='bold', color='black')
        plt.title('Bar Chart of Values Sorted by Rank')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.1)
        plt.tight_layout()
        # 显示条形图
        plt.show()
        break

    return 0


if __name__ == '__main__':
    main()
