import argparse
import json
import re
import os
import yaml
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, PeftModel
from loguru import logger
from transformers.generation import GenerationConfig
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from bertviz import model_view

column_name_json = '/public/whr/hzm/code/qwen2/ai_doctor/source/patient_infos_column_name_0_1_2.json'

generation_config_dir = '/public/whr/hzm/code/qwen2/ai_doctor/feature_select/'

cuda = "cuda:7"


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--dir-id", type=str, default="20240725-104805")
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
    attn_map = calc_attn_one_model(args, '20240813-193004')

    return 0


def calc_attn_one_model(args, dir_id):
    # sft_dir_id = '20240813-193004'  # '20240813-175203'  # '20240813-151014'  # '20240813-143537'  # '20240812-230700'
    # dpo_dir_id = '20240813-193004'  # '20240813-175203'  # '20240813-151014'  # '20240813-143537'  # '20240812-230700'  # '20240811-201031'
    diagnose_test_dataset_json = os.path.join(args.ft_path['dpo_model_dir'], dir_id, 'source',
                                              args.file_name['test_data'])
    diagnose_test_label_json = os.path.join(args.ft_path['dpo_model_dir'], dir_id, 'source',
                                            args.file_name['test_label'])
    base_model_dir = os.path.join(args.ft_path['sft_model_dir'], dir_id)
    dpo_model_dir = os.path.join(args.ft_path['dpo_model_dir'], dir_id)
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
    attn_map_f = {}
    total_len = len(label_info)
    for i in range(total_len):
        content = diagnose_test_dataset[i] + '\n' + args.prompt['finetune_diagnose_require']
        messages = [
            {"role": "system", "content": "You are an ophthalmology specialist."},
            {"role": "user", "content": content}
        ]
        if i == 0: logger.info(f'========= attention show messages: {messages}')
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

        output_text = tokenizer.decode(output['sequences'][0], skip_special_tokens=True)
        logger.info(f'output_text: {output_text}')

        inputs = tokenizer(output_text, return_tensors="pt").to(cuda)
        out = model(**inputs, output_attentions=True)
        attention = out['attentions']
        logger.info(f'attention shape: {attention[0].shape}')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        logger.info(f'tokens: {tokens}')
        model_view(attention, tokens)
        break
        #
        # input_ids_tensor = model_inputs["input_ids"]
        # input_ids_len = input_ids_tensor.shape[1]
        # input_ids = input_ids_tensor[0].tolist()
        #
        # # logger.info(f'input_ids: {input_ids}')
        #
        # # stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
        #
        # # output = model.generate(
        # #     **model_inputs,
        # #     # stop_words_ids=stop_words_ids,
        # #     generation_config=generation_config,
        # #     # max_new_tokens=100,
        # #     return_dict_in_generate=True,
        # #     output_attentions=True,
        # #     output_scores=True,
        # # )
        # # logger.info(f'-------- output type: {type(output)}')
        # # logger.info(f'-------- output sequences type: {type(output["sequences"])}')
        # # logger.info(f'-------- output sequences shape: {output["sequences"].shape}')
        # # logger.info(f'-------- output sequences[0] shape: {output["sequences"][0].shape}')
        # # logger.info(f'-------- output sequences[0]: {output["sequences"][0]}')
        # attentions = output["attentions"]
        # # logger.info(f'-------- output attention type :{attentions[-1][-1]}')
        # # logger.info(f'-------- output attention type :{type(attentions[-1][-1])}')
        # # logger.info(f'-------- output attention <0 count: {(attentions[-1][-1]<0).sum().item()}')
        #
        # # logger.info(f'-------- output attention length: {len(attentions)}')
        # # logger.info(f'-------- output attention [-1]: {attentions[-1]}')
        # # logger.info(f'-------- output attention [-1] length: {len(attentions[-1])}')  # 取输出的最后一个token对应的attentions
        # # logger.info(f'-------- output attention [-1][-1]: {attentions[-1][-1]}')  # 取输出的最后一个token对应的最后一层attention
        # # logger.info(
        # #     f'-------- output attention [-1][-1] shape: {attentions[-1][-1].shape}')  # 取输出的最后一个token对应的最后一层attention
        #
        # # tokens = tokenizer.tokenize(text)
        # tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor[0])
        # logger.info(f'tokens: {tokens}')
        #
        # logger.info(f'attentions[-1] len: {len(attentions[-1])}')
        # logger.info(f'attentions[-1] shape: {attentions[-1][0].shape}')
        # logger.info(f'tokens: {tokens}')
        # model_view(attentions[-1], tokens, include_layers=[1, -1], include_heads=[0, -1])

    return attn_map_f


if __name__ == '__main__':
    main()
