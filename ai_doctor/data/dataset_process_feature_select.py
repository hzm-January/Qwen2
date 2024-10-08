import argparse
import os
import json
import yaml
import uuid
import pandas as pd
from loguru import logger
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from note_template import NoteTemplate
from note_generator_helper import clean_query
from note_template_config import *


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class', type=str, default='Single')
    parser.add_argument('--config', type=str, default='/public/whr/hzm/code/qwen2/ai_doctor/config/dataset_config.yaml')
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

    return args


# def train_test_split(dataset):
#     # Don't want to shuffle bc done later with right seed to make it identical with external evaluation
#     data_train, data_test = train_test_split(data, test_size=0.20, shuffle=False)
#     data_valid, data_test = train_test_split(data_test, test_size=0.50, shuffle=False)
#     return data_train, data_valid, data_test

def note_template(args, dataset):
    train_dataset = dataset['train'].loc[:, dataset['train'].columns != 'label']
    train_labels = dataset['train']['label'].values.tolist()
    test_dataset = dataset['test'].loc[:, dataset['test'].columns != 'label']
    test_labels = dataset['test']['label'].values.tolist()
    train_notes = train_dataset.apply(lambda row: ', '.join(f"{c} is {row[c]}" for c in train_dataset.columns),
                                      axis=1).tolist()
    test_notes = test_dataset.apply(lambda row: ', '.join(f"{c} is {row[c]}" for c in test_dataset.columns),
                                    axis=1).tolist()

    # 1 sft
    sft_train_queries = []
    for i, note in enumerate(train_notes):
        sys_value = 'You are an ophthalmology specialist.'
        user_value = args.prompt['finetune_diagnose_prefix'] + '\n' + note + '\n' + args.prompt[
            'finetune_diagnose_require']
        ass_value = "yes" if train_labels[i] else "no"
        patient_description = {'type': 'chatml',
                               'source': 'self-made',
                               'messages': [{'role': 'system', 'content': sys_value},
                                            {'role': 'user', 'content': user_value},
                                            {'role': 'assistant', 'content': ass_value}],
                               }
        sft_train_queries.append(patient_description)

    # 2 dpo
    dpo_train_queries = []
    for i, note in enumerate(train_notes):
        sys_value = 'You are an ophthalmology specialist.'
        user_value = args.prompt['finetune_diagnose_prefix'] + '\n' + note + '\n' + args.prompt[
            'finetune_diagnose_require']
        label = train_labels[i]
        patient_description = {'type': 'chatml',
                               'id': str(uuid.uuid4()),
                               'chosen': [{"role": "system", "content": sys_value},
                                          {'role': 'user', 'content': user_value},
                                          {'role': 'assistant', 'content': 'yes' if label else 'no'}],
                               'rejected': [{"role": "system", "content": sys_value},
                                            {'role': 'user', 'content': user_value},
                                            {'role': 'assistant', 'content': 'no' if label else 'yes'}],
                               'prompt': user_value
                               }
        dpo_train_queries.append(patient_description)

    test_queries = [tq for tq in test_notes]  # add prompt

    # logger.info(f'train note : {train_note[0]}')
    # logger.info(f'test note : {test_note[0]}')

    logger.info(f'sft train query: {sft_train_queries[0]}')
    logger.info(f'dpo train query: {dpo_train_queries[0]}')
    logger.info(f'test query: {test_queries[0]}')

    return sft_train_queries, dpo_train_queries, test_queries, test_labels


def load_dataset(args):
    # TODO: 1 load excel
    df = pd.read_excel(os.path.join(args.path['dataset_dir'], args.file_name['org_data']), sheet_name=args.table_ids)
    # abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    # df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)

    # dataset = byte_to_string_columns(dataset)
    logger.info(f'Loaded dataset from {df}')
    logger.info(f'original columns count: {sum([len(df[sdf_n].columns) for sdf_n in list(df.keys())])}')

    # TODO: 2 preprocess
    df = preprocess(args, df)

    logger.info(f'df columns: {df.columns}')
    logger.info(f'columns count for delete duplicate: {df.shape[1]}')
    logger.info(f'dataset label count: 1 - {df["label"].sum()}, 0 - {df["label"].eq(0).sum()}')

    # TODO: 3 split dataset into train_dataset and test_dataset
    # dataset.rename(columns={'0= N 1=FFKC 2=SKC 3=CKC': 'label'}, inplace=True)
    # dataset['label'] = dataset['label'] != 0

    if args.tune_hyperparams:  # shuffle
        dataset_train, dataset_test = train_test_split(df, test_size=args.test_dataset_ratio, shuffle=args.shuffle,
                                                       random_state=args.seed, stratify=df['label'])
    else:  # shuffle & random split
        dataset_train, dataset_test = train_test_split(df, test_size=args.test_dataset_ratio, shuffle=args.shuffle,
                                                       stratify=df['label'])
    logger.info(
        f'train dataset label count: 1 - {dataset_train["label"].sum()}, 0 - {dataset_train["label"].eq(0).sum()}')
    logger.info(f'test dataset label count: 1 - {dataset_test["label"].sum()}, 0 - {dataset_test["label"].eq(0).sum()}')

    dataset = {'train': dataset_train, 'test': dataset_test}
    return dataset


def preprocess(args, df):
    df = preprocess_yd(args, df)
    df = preprocess_format(args, df)
    df = preprocess_finetune(args, df)
    df = preprocess_abbr(args, df)
    df = preprocess_feature_select(args, df)
    return df


def preprocess_feature_select(args, df):
    # df = df[[
    #     'Steepest point of the front surface keratometry displacement in the y-axis',
    #     'Dist. Apex-Thin.Loc. [mm](Dist. C-T)',
    #     'K1 B (D)',
    #     'K1 F (D)',
    #     'Root-mean-square of total aberrations of whole cornea',
    #     'BAD Dy',
    #     'Mean eccentricity in the central 30 degrees by Fourier analysis',
    #     'Steepest point of the front surface keratometry displacement in the x-axis',
    #     'Index of height asymmetry',
    #     'Maximum keratometry of the front surface',
    #     'BAD Dt',
    #     'Pachy Apex(CCT)',
    #     'Ambrósio’s relational thickness in the horizontal profile',
    #     'BAD Da',
    #     'index of vertical asymmetry',
    #     'RMS (CF)',
    #     'BAD Df',
    #     'Corneal volume in a 3mm diameter zone around the corneal apex',
    #     'K2 F (D)',
    #     'Pachy Prog Index Max.',
    #     'label'
    # ]]
    df = df[[
        "性别",
        "幼年时家庭经济状况",
        "BMI",
        "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？",
        "年龄",
        "圆锥角膜家族史",
        "文化程度",
        "Dist. Apex-Thin.Loc. [mm](Dist. C-T)",
        "Ambrósio’s relational thickness in the horizontal profile",
        "Pachy Prog Index Min.",
        "每天使用电子屏幕（手机、电脑等）的总时间（小时）",
        "揉眼睛的频率",
        "Steepest point of the front surface keratometry displacement in the x-axis",
        "Pachy Apex(CCT)",
        "每天在黑暗环境中使用电子屏幕的时间（小时）",
        "是否患有过敏性疾病？",
        "感到工作/学习压力很大？",
        "Mean eccentricity in the central 30 degrees by Fourier analysis",
        "睡觉时是否偏好把手或手臂垫放在眼睛上？",
        "Root-mean-square of total aberrations of whole cornea"
    ]]
    return df


def preprocess_yd(args, df):
    # rename label column as label_i in each sheet
    handle_columns = []
    for i, df_name in enumerate(df.keys()):
        print(i, df_name)
        handle_columns.append(f'label_{i}')
        df[df_name].rename(columns={args.label_column_name: handle_columns[i]}, inplace=True)

    # delete duplicate columns
    unique_columns = []
    for i, df_name in enumerate(df.keys()):
        for col in df[df_name].columns:
            if col not in unique_columns:
                unique_columns.append(col)
            else:
                df = df.drop(columns=[col])

    # concat sheets by column
    max_rows = max(df.shape[0] for df in df.values())
    # complement rows and concat all sheets in this excel
    df = pd.concat(
        [sdf.reindex(range(max_rows)) for sdf in df.values()],
        axis=1
    )

    # generate label for binary_class or multi_class
    if args.binary_class:
        df['label'] = df[handle_columns].sum(axis=1).apply(lambda x: 1 if x != 0 else 0)
    else:  # multi_class
        df['label'] = df[handle_columns].apply(lambda row: mode(row, nan_policy='omit').mode[0], axis=1)
    df.drop(columns=handle_columns, inplace=True)

    return df


def preprocess_format(args, df):
    # data format process
    df[df.select_dtypes(['object']).columns] = df.select_dtypes(['object']).apply(lambda x: x.str.strip())

    # pattern = re.compile('|'.join(map(re.escape, replacements.keys())))

    # , '-': ' '
    df.columns = [col.translate(str.maketrans({':': '', '：': ''})) for col in df.columns]
    return df


def preprocess_abbr(args, df):
    with open(os.path.join(args.path['dataset_dir'], args.file_name['abbr_mapping']), 'r') as f:
        abbr_map = json.load(f)

    logger.info(f'abbr mapping: {abbr_map}')
    df.columns = [abbr_map.get(col, col) if col in abbr_map and abbr_map[col].strip() else col for col in df.columns]

    return df


def preprocess_finetune(args, df):
    # df = df[[c for c in df if df[c].nunique() > 1]]  # remove constant
    print(rule_yiduo)
    print(df.columns.values.tolist())
    # substitute digit to word
    for k, v in rule_yiduo.items():
        if (k not in df.columns.values.tolist()) or (not v): continue
        df[k] = pd.cut(df[k], bins=v["bins"], labels=v["labels"], right=False, include_lowest=False)
    return df


def main():
    # 1 config
    args = load_config()
    # logger.info(f'Processing {args}')
    logger.info(f'table_ids {args.table_ids}')
    logger.info(f'shuffle {args.shuffle} {type(args.shuffle)}')

    # 2 lode data
    dataset = load_dataset(args)

    # 3 template
    sft_train_queries, dpo_train_queries, test_queries, test_labels = note_template(args, dataset)

    with open(os.path.join(args.path['dataset_dir'], args.file_name['sft_fs_train_data']), 'w') as f:
        for row in sft_train_queries:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(sft_train_queries, f, ensure_ascii=False) # json
    with open(os.path.join(args.path['dataset_dir'], args.file_name['dpo_fs_train_data']), 'w') as f:
        for row in dpo_train_queries:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(dpo_train_queries, f, ensure_ascii=False)
    with open(os.path.join(args.path['dataset_dir'], args.file_name['test_fs_data']), 'w') as f:
        for row in test_queries:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(test_queries, f, ensure_ascii=False)
    with open(os.path.join(args.path['dataset_dir'], args.file_name['test_fs_label']), 'w') as f:
        for row in test_labels:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(test_labels, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
