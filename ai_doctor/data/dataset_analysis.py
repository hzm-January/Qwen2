import argparse
import os
import json
import yaml
import uuid
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from note_template_config import *

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
    df = pd.read_excel(os.path.join(args.path['dataset_dir'], args.file_name['org_data']), sheet_name=args.table_ids, dtype=special_column_type)
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


def load_dataset_cls(args):
    df = pd.read_excel(os.path.join(args.path['dataset_dir'], args.file_name['org_data']), sheet_name=args.table_ids,
                       dtype=special_column_type)
    index = 1
    df = df[index]

    # 指定用于分类的列名（例如“类别”列）
    category_col = '0= N 1=FFKC 2=SKC 3=CKC'

    # 确保类别列中只有 "正常" 和 "异常" 两种类别（可以根据实际数据修改）
    normal_data = df[df[category_col] == 0]  # 选择类别为正常的数据
    abnormal_data = df[df[category_col] == 1]  # 选择类别为异常的数据
    plt.style.use('seaborn-v0_8-dark')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 设置图表的大小
    # fig, axes = plt.subplots(len(df.columns) - 1, 1, figsize=(12, 6 * (len(df.columns) - 1)))
    fig, axes = plt.subplots(len(df.columns)-1, 1, figsize=(12, 6 * (len(df.columns)-1)))

    # 遍历除"类别"列外的所有列，并绘制每列的正常性和异常性数据的直方图
    for i, col in enumerate(df.columns):
        if col != category_col:
            ax = axes[i-1] if len(df.columns) - 1 > 1 else axes  # 处理只有一个子图的情况
            # 计算相同的 bins 范围
            min_bin = min(normal_data[col].min(), abnormal_data[col].min())
            max_bin = max(normal_data[col].max(), abnormal_data[col].max())
            bins = np.linspace(min_bin, max_bin, 60)  # 设置 20 个区间
            # 绘制正常性数据直方图
            normal_data[col].hist(bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='normal', ax=ax)
            # 绘制异常性数据直方图
            abnormal_data[col].hist(bins=bins, color='lightcoral', edgecolor='black', alpha=0.5, label='abnormal', ax=ax)

            # 计算男性数据的均值和方差
            normal_mean = normal_data[col].mean()
            normal_std = normal_data[col].std()

            # 计算女性数据的均值和方差
            abnormal_mean = abnormal_data[col].mean()
            abnormal_std = abnormal_data[col].std()

            # 添加均值和方差到图表上
            ax.text(0.02, 0.95, f'正常 - 均值: {normal_mean:.2f}, 方差: {normal_std:.2f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', color='blue')
            ax.text(0.02, 0.90, f'异常 - 均值: {abnormal_mean:.2f}, 方差: {abnormal_std:.2f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', color='red')

            # 添加图例、标题和坐标轴标签
            ax.set_title(f'Distribution of {col} by label', fontsize=16)
            ax.set_xlabel('Value', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.legend()

    # 添加总标题
    plt.suptitle('Data Distribution by Label', fontsize=20)

    # 调整布局，防止图表重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 显示图表
    # plt.show()

    # 保存图像到指定路径
    plt.savefig(f'/data/whr/hzm/file/yiduo/dataset_analysis_ffkc_table_{index}.png', dpi=200)  # 指定文件路径和分辨率


def main():
    # 1 config
    args = load_config()
    # logger.info(f'Processing {args}')
    logger.info(f'table_ids {args.table_ids}')
    logger.info(f'shuffle {args.shuffle} {type(args.shuffle)}')

    # 2 lode data
    dataset = load_dataset_cls(args)

if __name__ == '__main__':
    main()