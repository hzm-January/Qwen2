import argparse
import os
import json
import yaml
import uuid
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, default='single')
    parser.add_argument('--selected', type=int, default=1)
    parser.add_argument('--digit-to-word', type=int, default=1)
    parser.add_argument('--config', type=str, default='/data/whr/hzm/code/qwen2/ai_doctor/config/dataset_config.yaml')
    parser.add_argument('--output_dir', type=str, default='Single')
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
    logger.info('=================== config =================')
    logger.info(args)
    return args


def load_dataset(args):
    # TODO: 1 load excel
    df = pd.read_excel(os.path.join(args.path['dataset_dir'], args.file_name['org_data']), sheet_name=args.table_ids)
    # abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    # df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)

    # dataset = byte_to_string_columns(dataset)
    # logger.info(f'Loaded dataset from {df}')
    logger.info(f'original columns count: {sum([len(df[sdf_n].columns) for sdf_n in list(df.keys())])}')
    print(plt.style.available)
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-dark')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # 绘制每列的柱状图
    df[0].hist(bins=10, figsize=(20, 12), color='skyblue', edgecolor='black')

    # 添加总标题
    plt.suptitle('Data Distribution for Each Column', fontsize=16)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 显示图表
    plt.show()



def main():
    # 1 config
    args = load_config()
    # logger.info(f'Processing {args}')
    logger.info(f'table_ids {args.table_ids}')
    logger.info(f'shuffle {args.shuffle} {type(args.shuffle)}')

    # 2 lode data
    dataset = load_dataset(args)

if __name__ == '__main__':
    main()