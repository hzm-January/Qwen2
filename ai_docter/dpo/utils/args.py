from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum


# 目前支持的template 类型
class TemplateName(Enum):
    QWEN = 'qwen'
    YI = 'yi'
    GEMMA = 'gemma'
    PHI_3 = 'phi-3'
    DEEPSEEK = 'deepseek'
    MINICPM = 'minicpm'
    LLAMA2 = 'llama2'
    LLAMA3 = 'llama3'


class TrainMode(Enum):
    QLORA = 'qlora'
    LORA = 'lora'
    FULL = 'full'


class TrainArgPath(Enum):
    SFT_LORA_QLORA_BASE = 'train_args/sft/lora_qlora/base.py'
    DPO_LORA_QLORA_BASE = 'train_args/dpo/dpo_config.py'
    TRAIN_DATASET_PATH = '/data1/llm/houzm/99-code/01-Qwen-VL/ai_doctor/data/data_finetune/dpo/dpo_finetune_dataset.jsonl'
    # TRAIN_DATASET_PATH = '/data1/llm/houzm/99-code/03-qwen-dpo/data/dpo_multi_data.jsonl'
    MODLE_PATH = '/data1/llm/houzm/98-model/01-qwen-vl-chat/qwen/qwen-dpo/input-model/v2'
    TASK_TYPE = 'dpo_multi'


@dataclass
class CommonArgs:
    """
    一些常用的自定义参数
    """
    # Deepspeed相关参数
    local_rank: int = field(default=1, metadata={"help": "deepspeed所需参数,单机无需修改"})
    # max_memory: dict = field(default_factory={0:"40GiB",2:"40GiB",3:"40GiB",4:"40GiB"})

    train_args_path: TrainArgPath = field(default=TrainArgPath.DPO_LORA_QLORA_BASE.value,
                                          metadata={"help": "当前模式的训练参数,分为sft和dpo参数"})
    max_len: int = field(default=4096, metadata={"help": "最大输入长度,dpo时该参数在dpo_config中设置"})
    max_prompt_length: int = field(default=4096, metadata={
        "help": "dpo时，prompt的最大长度，适用于dpo_single,dpo_multi时该参数在dpo_config中设置"})
    train_data_path: Optional[str] = field(default=TrainArgPath.TRAIN_DATASET_PATH.value, metadata={"help": "训练集路径"})
    model_name_or_path: str = field(default=TrainArgPath.MODLE_PATH.value, metadata={"help": "下载的所需模型路径"})
    template_name: TemplateName = field(default=TemplateName.QWEN.value,
                                        metadata={"help": "sft时的数据格式,即指定模型数据输入格式"})

    # 微调方法相关选择与配置
    train_mode: TrainMode = field(default=TrainMode.LORA.value,
                                  metadata={"help": "选择采用的训练方式：[qlora, lora, full]"})
    use_dora: bool = field(default=False, metadata={"help": "仅在train_mode==lora时可以使用。是否使用Dora(一个基于lora的变体) "
                                                            "目前只支持linear and Conv2D layers."})

    task_type: str = field(default=TrainArgPath.TASK_TYPE.value,
                           metadata={"help": "预训练任务：[pretrain, sft, dpo_multi, dpo_single]，目前支持sft,dpo"})

    # lora相关配置
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
