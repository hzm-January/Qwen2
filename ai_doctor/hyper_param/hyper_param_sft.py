import optuna
from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, HfArgumentParser
import torch
import transformers
import argparse
from examples.sft.finetune import *
from ai_doctor.test.qwen2_sft_hyper_param_test import *


def generate_learning_rate():
    return [1e-4, 1e-5, 1e-6]


def generate_batch_size():
    return [1, 2, 3, 4]


def generate_train_epochs():
    return [1, 2, 3, 4, 5, 6, 7]


def generate_gradient_accumulation_steps():
    return [1, 2, 3, 4]


# @dataclass
# class SFTTrainingArguments(TrainingArguments):
#     learning_rate: list[float] = field(default_factory=generate_learning_rate,
#                                        metadata={"help": "learning rate options"})
#     per_device_train_batch_size: list[int] = field(default_factory=generate_batch_size,
#                                                    metadata={"help": "per_device_train_batch_size options"})
#     num_train_epochs: list[int] = field(default_factory=generate_train_epochs,
#                                         metadata={"help": "train epochs options"})
#     gradient_accumulation_steps: list[int] = field(default_factory=generate_gradient_accumulation_steps,
#                                                    metadata={"help": "generate_gradient_accumulation_steps options"})


# 定义模型训练函数
def train_model(trial):

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # 定义超参数的搜索空间
    training_args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    training_args.per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [1, 2, 3])
    training_args.num_train_epochs = trial.suggest_int('num_train_epochs', 1, 10)
    training_args.gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 3)

    logger.info(f"============== train args: {training_args}")
    logger.info(f"============== data args: {data_args}")
    logger.info(f"============== model args: {model_args}")
    logger.info(f"============== lora args: {lora_args}")

    # # 定义超参数的搜索空间
    # options_args.learning_rate_options = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    # options_args.per_device_train_batch_size_options = trial.suggest_categorical('per_device_train_batch_size',
    #                                                                              [1, 2, 3, 4, 5, 6, 7, 8])
    # options_args.num_train_epochs_options = trial.suggest_int('num_train_epochs', 1, 10)
    # options_args.gradient_accumulation_steps_options = trial.suggest_int('gradient_accumulation_steps', 1, 10)

    # 假设数据加载器和模型已经定义好了
    # train_dataloader, eval_dataloader, model = ...

    # 定义训练参数
    # training_args = TrainingArguments(
    #     output_dir='./results',
    #     learning_rate=learning_rate,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     num_train_epochs=train_epochs,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     logging_dir='./logs',
    #     load_best_model_at_end=True,
    # )

    # 使用 Transformers 的 Trainer 进行训练

    # 训练模型并评估
    train(model_args, data_args, training_args, lora_args)

    # 评估验证集性能，假设评估函数返回一个包含准确率的字典
    eval_results = eval()
    accuracy = eval_results.get('eval_accuracy', 0)

    # 目标是最大化验证集准确率，返回负的准确率以供优化器最小化
    return accuracy


# 使用 Optuna 进行贝叶斯优化
def objective(trial):
    return train_model(trial)


# def load_config():
#     # parser = transformers.HfArgumentParser(TrainingArguments)
#     # train_args = parser.parse_args_into_dataclasses()[0]
#
#     # parser = argparse.ArgumentParser()
#     # # parser.add_argument('--learning_rate_options', type=float, nargs=3, default=[1e-4, 1e-5, 1e-6])
#     # # parser.add_argument('--per_device_train_batch_size_options', type=int, nargs=4, default=[1, 2, 3, 4])
#     # # parser.add_argument('--num_train_epochs_options', type=int, nargs=7, default=[1, 2, 3, 4, 5, 6, 7])
#     # # parser.add_argument('--gradient_accumulation_steps_options', type=int, nargs=4, default=[1, 2, 3, 4])
#     # parser.add_argument('--learning_rate_options', type=float, default=1e-5)
#     # parser.add_argument('--per_device_train_batch_size_options', type=int, default=1)
#     # parser.add_argument('--num_train_epochs_options', type=int, default=3)
#     # parser.add_argument('--gradient_accumulation_steps_options', type=int, default=1)
#     # options_args = parser.parse_args()
#
#
#
#     return options_args


def main():
    # options_args = load_config()

    # create Optuna object
    study = optuna.create_study(direction="maximize")

    # optimize
    # study.optimize(lambda trial: objective(trial, options_args), n_trials=20)  # 进行 20 次超参数优化
    study.optimize(objective, n_trials=20)  # 进行 20 次超参数优化

    # print best params
    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print("  Best hyperparameters: ", trial.params)

    return 0


if __name__ == '__main__':
    main()
