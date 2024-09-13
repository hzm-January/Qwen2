import os, torch, sys
sys.path.append('/data/whr/hzm/code/qwen2/')

'''逻辑：先复制Qwen_VL并手动更改模型配置与网络结构(Qwen_VL_tmp)，再从各种路径加载参数，保存为最新的离线模型（Qwen_VL_new）'''

#修改config文件
import copy, math
vl_model_path = '/data/whr/hzm/model/qwen2-base/qwen2/qwen2-7b-instruct' # 原始模型
output_path = '/data/whr/hzm/model/qwen2-extra/qwen2/qwen2-7b-instruct-extra' # 新模型路径

from functools import partial
#TODO: 修改前，曹莉莉代码
# from Qwen_VL_tmp.modeling_qwen import QWenLMHeadModel
# from Qwen_VL_tmp.tokenization_qwen import QWenTokenizer
# from Qwen_VL_tmp.configuration_qwen import QWenConfig
#TODO：修改后，侯志明代码
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from torch import nn


# num_embeddings: 151936 embedding_dim: 4096
# 151648
#TODO: 修改前 曹莉莉代码
# json_embed = nn.Embedding(151936, 4096)  # 新增对json format数据的编码器
#TODO: 修改后 侯志明代码
json_embed = nn.Embedding(152064, 3584)  # 新增对json format数据的编码器


json_embed.weight.data.normal_(mean=0.0, std=0.02)

print(111111111, torch.mean(json_embed.weight))
print(111111111, torch.var(json_embed.weight))

# text_model_path配置更改完毕后，再加载模型，就包含了更改后的模型结构。
qwen_vl_model = Qwen2ForCausalLM.from_pretrained(vl_model_path, trust_remote_code=True, device_map='auto', force_download=True)

config = Qwen2Config.from_pretrained(
        vl_model_path, trust_remote_code=True, force_download=True
    )
tokenizer = Qwen2Tokenizer.from_pretrained(vl_model_path, trust_remote_code=True, force_download=True)

# 一定要继承wte的参数，不然效果很差！
#TODO: 修改前，曹莉莉代码
# qwen_vl_model.transformer.json_embed = copy.deepcopy(qwen_vl_model.transformer.wte)
#TODO: 修改后，侯志明代码
qwen_vl_model.model.json_embed = copy.deepcopy(qwen_vl_model.model.embed_tokens)

print(22222, torch.mean(qwen_vl_model.model.json_embed.weight))
print(22222, torch.var(qwen_vl_model.model.json_embed.weight))
print(33333, torch.mean(qwen_vl_model.model.embed_tokens.weight))
print(33333, torch.var(qwen_vl_model.model.embed_tokens.weight))


model=copy.deepcopy(qwen_vl_model)


model.save_pretrained(output_path, max_shard_size="2048MB", safe_serialization=True)
config.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
# todo: 手动将py文件copy到新路径下
# cp *.py ../Qwen_VL_new/