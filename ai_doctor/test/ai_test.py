from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/whr/hzm/model/qwen2-base/qwen2/qwen2-7b-instruct")
token = tokenizer('\<|extra_0|\>')
print(token)