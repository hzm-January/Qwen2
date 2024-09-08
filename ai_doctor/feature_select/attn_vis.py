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

import seaborn as sns
from matplotlib.colors import LogNorm

column_name_json = '/data/whr/hzm/code/qwen2/ai_doctor/source/patient_infos_column_name_0_1_2.json'

generation_config_dir = '/data/whr/hzm/code/qwen2/ai_doctor/feature_select/'

cuda = "cuda:7"


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--dir-id", type=str, default="20240828-123208")
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
    args = load_config()
    attn_map = calc_attn_one_model(args, '20240828-123208')

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
        prompt = args.prompt['finetune_diagnose_require']
        if args.cls == 'multiple': prompt = args.prompt['finetune_diagnose_require_mc']
        content = diagnose_test_dataset[i] + '\n' + prompt
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

        input_ids_tensor = model_inputs["input_ids"]
        input_ids_len = input_ids_tensor.shape[1]
        input_ids = input_ids_tensor[0].tolist()

        attentions = output["attentions"]
        attentions = torch.stack(attentions[-1]).to(cuda)
        attention_layers = torch.mean(attentions, axis=2).float()
        layer_num, bs, s1, s2 = attention_layers.shape
        attention_layers = attention_layers.reshape(layer_num, s1, s2)
        output_token_decode_str = tokenizer.decode(output["sequences"][0], skip_special_tokens=True)

        answer = output_token_decode_str[len(content) - 1:]

        # layer_num = attention_layers.shape[0]

        attn_map_layer = []

        for l in range(layer_num):

            attention = attention_layers[l]

            attn_map = {}

            for i, column_name in enumerate(column_names):
                if 'Stress' in column_name: continue  # TODO: 待修复，Stress-strain 匹配不到
                cur_strs = re.findall(fr'({re.escape(column_name)}.*?),', content)
                # logger.info(f'{i}----{column_name}-----{cur_strs}')
                if len(cur_strs) == 0: continue
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
                # print(start_i)
                attn = attention[-1][start_i: start_i + cur_str_ids_len]
                # logger.info(f'-------- attn: {attn}')
                attn_sum = torch.sum(attn).item()
                attn_map[column_name] = attn_sum

            attn_map_layer.append(list(attn_map.values()))
            ## rank attentions
            # sorted_items = sorted(attn_map.items(), key=lambda item: item[1], reverse=False)
            # sorted_attn_map = {key: value for key, value in sorted_items}
            # # logger.info(f'-------- attention: {sorted_attn_map}')
            #
            # # keys, values = zip(*sorted_items[:30])
            # keys, values = zip(*sorted_items)
            #
            # # show_attn_rank(keys, values)

        attns_for_columns = torch.tensor(attn_map_layer).to(cuda)

        layer = 0
        # visualize_attention(attentions[-1][layer],
        #                     output_path=args.path["dataset_dir"] + f"/attn_maps/atten_map_{layer}.png",
        #                     title=f"layer {layer}")

        visualize_attention(attns_for_columns,
                            output_path=args.path["dataset_dir"] + f"/attn_maps/atten_map_{layer}.png",
                            title=f"layer {layer}")

        break

    return attn_map_f


def visualize_attention(averaged_attention, output_path="atten_map_1.png", title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    # averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)

    # pooling the attention scores  with stride 20
    # averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20,
    #                                                     stride=20).squeeze(0).squeeze(0)

    cmap = plt.colormaps.get_cmap("viridis")
    plt.figure(figsize=(5, 5), dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # set the x and y ticks to 20x of the original

    ax = sns.heatmap(averaged_attention.cpu(),
                     cmap=cmap,  # custom color map
                     norm=log_norm,  #
                     # cbar_kws={'label': 'Attention score'},
                     )

    # remove the x and y ticks

    # replace the x and y ticks with string

    x_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[1])]
    y_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0, averaged_attention.shape[1])])
    ax.set_yticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # change the x tinks font size
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)

    # make y label vertical
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.title(title)
    # tight layout
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()

    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)

    return top_five_attentions, averaged_attention


if __name__ == '__main__':
    main()
