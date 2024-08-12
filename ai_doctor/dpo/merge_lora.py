from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# base模型和lora训练后保存模型的位置
base_model_path = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat'
lora_path = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/Qwen-VL-Chat/hzm_qwen_finetune/diagnose/20240807-212707/'
# 合并后整个模型的保存地址
merge_output_dir = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/input-model/v2'

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cuda:0",
    torch_dtype="auto",
    trust_remote_code=True,
    bf16=True
)

lora_model = PeftModel.from_pretrained(base_model, lora_path)
model = lora_model.merge_and_unload()

if merge_output_dir:
    model.save_pretrained(merge_output_dir)
    tokenizer.save_pretrained(merge_output_dir)
