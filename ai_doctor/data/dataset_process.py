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
import random


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', type=str, default='single')
    parser.add_argument('--selected', type=int, default=1)
    parser.add_argument('--digit-to-word', type=int, default=0)
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


# def train_test_split(dataset):
#     # Don't want to shuffle bc done later with right seed to make it identical with external evaluation
#     data_train, data_test = train_test_split(data, test_size=0.20, shuffle=False)
#     data_valid, data_test = train_test_split(data_test, test_size=0.50, shuffle=False)
#     return data_train, data_valid, data_test

def note_template(args, dataset):
    dataset['train'] = dataset['train'][dataset['train']['label'].isin(args.train_labels)]
    train_dataset = dataset['train'].loc[:, dataset['train'].columns != 'label']
    train_labels = dataset['train']['label'].values.tolist()
    # 删除label中为 2，3 的数据，2024.09.03，训练使用label2,3测试不需要使用label2,3的数据
    dataset['test'] = dataset['test'][dataset['test']['label'].isin(args.test_labels)]
    test_dataset = dataset['test'].loc[:, dataset['test'].columns != 'label']
    test_labels = dataset['test']['label'].values.tolist()
    train_notes = train_dataset.apply(lambda row: ' '.join(f"{c} is {row[c]}" for c in train_dataset.columns),
                                      axis=1).tolist()
    test_notes = test_dataset.apply(lambda row: ' '.join(f"{c} is {row[c]}" for c in test_dataset.columns),
                                    axis=1).tolist()

    # 1 sft
    sft_train_queries = []
    for i, note in enumerate(train_notes):
        sys_value = 'You are an ophthalmology specialist.'
        prompt = args.prompt['finetune_diagnose_require']
        if args.cls.lower() == 'multiple': prompt = args.prompt['finetune_diagnose_require_mc']
        user_value = args.prompt['finetune_diagnose_prefix'] + '\n' + note + '\n' + prompt
        ass_value = generate_ass_value_by_label(train_labels[i], args, 'pos')
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
        prompt = args.prompt['finetune_diagnose_require']
        if args.cls.lower() == 'multiple': prompt = args.prompt['finetune_diagnose_require_mc']
        user_value = args.prompt['finetune_diagnose_prefix'] + '\n' + note + '\n' + prompt
        ass_value_pos = generate_ass_value_by_label(train_labels[i], args, 'pos')
        ass_value_neg = generate_ass_value_by_label(train_labels[i], args, 'neg')
        patient_description = {'type': 'chatml',
                               'id': str(uuid.uuid4()),
                               'chosen': [{"role": "system", "content": sys_value},
                                          {'role': 'user', 'content': user_value},
                                          {'role': 'assistant', 'content': ass_value_pos}],
                               'rejected': [{"role": "system", "content": sys_value},
                                            {'role': 'user', 'content': user_value},
                                            {'role': 'assistant', 'content': ass_value_neg}],
                               'prompt': user_value
                               }
        dpo_train_queries.append(patient_description)

    test_queries = [tq for tq in test_notes]  # add prompt

    # logger.info(f'train note : {train_note[0]}')
    # logger.info(f'test note : {test_note[0]}')

    logger.info(f'===== after preprocess train dataset label count: {train_labels.count(0)},'
                f'{train_labels.count(1)}, '
                f'{train_labels.count(2)}, '
                f'{train_labels.count(3)}')
    logger.info(f'===== after preprocess test dataset label count: {test_labels.count(0)},'
                f'{test_labels.count(1)}, '
                f'{test_labels.count(2)}, '
                f'{test_labels.count(3)}')

    logger.info(f'sft train query: {sft_train_queries[0]}')
    logger.info(f'dpo train query: {dpo_train_queries[0]}')
    logger.info(f'test query: {test_queries[0]}')
    logger.info(f'test label: {test_labels[0]}')

    return sft_train_queries, dpo_train_queries, test_queries, test_labels


def generate_ass_value_by_label(label, args, flag):
    label_words = ["No", "forme fruste keratoconus", "subclinical keratoconus", "clinical keratoconus"]
    if flag == 'pos':  # positive
        ass_value = "yes" if label else "no"
        if args.cls.lower() == 'multiple':
            ass_value = label_words[label]
    else:
        ass_value = "no" if label else "yes"
        if args.cls.lower() == 'multiple':
            ass_value = label_words[random.choice(list(set(args.train_labels) - {label}))]
    return ass_value


def load_dataset(args):
    # TODO: 1 load excel
    df = pd.read_excel(os.path.join(args.path['dataset_dir'], args.file_name['org_data']), sheet_name=args.table_ids)
    # abbr_map = dict(zip(am_df[am_df.columns[0]], am_df[am_df.columns[1]]))
    # df = pd.read_excel(load_path + file_names['org_data_xlsx'], sheet_name=None)

    # dataset = byte_to_string_columns(dataset)
    # logger.info(f'Loaded dataset from {df}')
    logger.info(f'original columns count: {sum([len(df[sdf_n].columns) for sdf_n in list(df.keys())])}')

    # TODO: 2 preprocess
    df = preprocess(args, df)

    # logger.info(f'df columns: {df.columns}')
    # logger.info(f'columns count for delete duplicate: {df.shape[1]}')
    # logger.info(f'dataset label count: 1 - {df["label"].sum()}, 0 - {df["label"].eq(0).sum()}')

    # TODO: 3 split dataset into train_dataset and test_dataset
    # dataset.rename(columns={'0= N 1=FFKC 2=SKC 3=CKC': 'label'}, inplace=True)
    # dataset['label'] = dataset['label'] != 0

    if args.tune_hyperparams:  # shuffle
        logger.info(f'============== dataset split seed: {args.seed} ==============')
        dataset_train, dataset_test = train_test_split(df, test_size=args.test_dataset_ratio, shuffle=args.shuffle,
                                                       random_state=args.seed, stratify=df['label'])
    else:  # shuffle & random split
        dataset_train, dataset_test = train_test_split(df, test_size=args.test_dataset_ratio, shuffle=args.shuffle,
                                                       stratify=df['label'])
    logger.info(f'===== befor preprocess train dataset label count: {dataset_train["label"].eq(0).sum()},'
                f'{dataset_train["label"].eq(1).sum()}, '
                f'{dataset_train["label"].eq(2).sum()}, '
                f'{dataset_train["label"].eq(3).sum()}')
    logger.info(f'===== befor preprocess test dataset label count: 0 - {dataset_test["label"].eq(0).sum()},'
                f'{dataset_test["label"].eq(1).sum()}, '
                f'{dataset_test["label"].eq(2).sum()}, '
                f'{dataset_test["label"].eq(3).sum()}')

    dataset = {'train': dataset_train, 'test': dataset_test}
    return dataset


def preprocess(args, df):
    df = preprocess_yd(args, df)
    df = preprocess_format(args, df)
    if args.digit_to_word:
        logger.info('====== digit to word ======')
        df = preprocess_d2w(args, df)
    else:
        df = preprocess_digit(args, df)
    df = preprocess_abbr(args, df)
    df = preprocess_finetune(args, df)
    if args.selected:
        logger.info('====== selected features ======')
        df = preprocess_feature_select(args, df)
    return df


def preprocess_feature_select(args, df):
    # top 10 keys
    # top_k = ["label", 'BMI', '性别', '揉眼睛的频率', '年龄', '睡觉时是否打鼾或患有睡眠呼吸暂停综合征？',
    #          '每天使用电子屏幕（手机、电脑等）的总时间（小时）', '惯用手', '幼年时家庭经济状况',
    #          '每次揉眼持续时间', '每天在黑暗环境中使用电子屏幕的时间（小时）', '文化程度', '春季角结膜炎',
    #          '最常采用的睡姿', '感到工作/学习压力很大？', '每天在户外阳光/紫外线下活动时间（小时）',
    #          '揉眼姿势', '阅读书籍', '揉眼时的力度', '是否怀过孕？', '常于夜间工作/学习？', '是否吸烟？',
    #          '职业', '是否患有过敏性疾病？', '最常揉眼的部位', '睡觉时是否偏好把手或手臂垫放在眼睛上？',
    #          '常在大量灰尘环境中工作或生活？', '是否饮酒？', '幼年时居住地', '是否对某些物质过敏？',
    #          '过敏性结膜炎', '眼睑松弛综合征', '甲状腺疾病', '是否用过外源性性激素药物？', '是否患有其他疾病？', '倒睫', '干眼症']
    # top 80 keys
    # top_k = [
    #     "label",
    #     "性别",
    #     "BMI",
    #     "Dist. Apex-Thin.Loc. [mm](Dist. C-T)",
    #     "揉眼睛的频率",
    #     "幼年时家庭经济状况",
    #     "Pachy Apex(CCT)",
    #     "文化程度",
    #     "Pachy Min (TCT)",
    #     "Ambrósio’s relational thickness in the horizontal profile",
    #     "Y position of the thinnest point",
    #     "Minimum Ambrósio relational thickness",
    #     "Average Ambrósio relational thickness",
    #     "X position of the thinnest point",
    #     "Mean eccentricity in the central 30 degrees by Fourier analysis",
    #     "Maximum keratometry of the front surface",
    #     "Km B (mm)",
    #     "Intraocular pressure",
    #     "幼年时居住地",
    #     "Biomechanically corrected intraocular pressure",
    #     "BAD Df",
    #     "Root-mean-square of total aberrations of whole cornea",
    #     "Index of surface variance",
    #     "Maximum Ambrósio relational thickness",
    #     "K1 F (D)",
    #     "Pachy Prog Index Min.",
    #     "RMS HOA (CB)",
    #     "RMS (CF)",
    #     "Steepest point of the front surface keratometry displacement in the x-axis",
    #     "每次揉眼持续时间",
    #     "Pachy Prog Index Max.",
    #     "Keratoconus index",
    #     "Corneal volume in a 5mm diameter zone around the corneal apex",
    #     "RMS LOA (CB)",
    #     "RMS HOA (CF)",
    #     "Pachy Prog Index Avg.",
    #     "Steepest point of the front surface keratometry displacement in the y-axis",
    #     "Change in Arclength during the second applanation moment from the initial state",
    #     "Corneal volume in a 10mm diameter zone around the corneal apex",
    #     "Stiffness parameter at the first applanation",
    #     "index of vertical asymmetry",
    #     "RMS (CB)",
    #     "Corneal volume in a 3mm diameter zone around the corneal apex",
    #     "Ratio between the central deformation and the average of the peripheral deformation determined at 1mm",
    #     "Elevation back",
    #     "K1 B (D)",
    #     "BAD Dp",
    #     "C.Volume(chamber volume)",
    #     "RMS LOA (CF)",
    #     "Change in Arclength during the first applanation moment from the initial state",
    #     "Elevation front",
    #     "Change in Arclength during the highest concavity moment from the initial state",
    #     "揉眼时的力度",
    #     "Highest concavity deflection length",
    #     "Integrated radius",
    #     "Maximum deformation amplitude",
    #     "BAD Dy",
    #     "BAD Db",
    #     "Maximum inverse concave radius",
    #     "Length at the the first applanation",
    #     "Root-mean-square of higher-order aberrations of whole cornea",
    #     "K2 F (D)",
    #     "Km F (D)",
    #     "Time of reaching the first applanation",
    #     "Index of height decentration",
    #     "Time of undergoing the greatest degree of deformation and reaching the highest concavity",
    #     "K2 B (D)",
    #     "Corneal volume in a 7mm diameter zone around the corneal apex",
    #     "Deflection area between the initial convex cornea and cornea at the first applanation on the analyzed horizontal sectional plane",
    #     "Index of height asymmetry",
    #     "Distance between the two bending peaks created in the cornea at the highest concavity",
    #     "Ratio between the central deformation and the average of the peripheral deformation determined at 2mm",
    #     "BAD Da",
    #     "Deflection amplitude of the corneal apex at the first applanation",
    #     "Deflection amplitude of the corneal apex at the highest concavity",
    #     "BAD Dt",
    #     "Length at the the second applanation",
    #     "Chamber angle",
    #     "Deflection area between the initial convex cornea and cornea at the highest concavity on the analyzed horizontal sectional plane",
    #     "Time of reaching the second applanation",
    #     "A.C.Depth Int"
    # ]
    # df = df[top_k[:11]]  # include label
    # top 60 keys
    top_k = [
        "label",
        "性别",
        "幼年时家庭经济状况",
        "BMI",
        "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？",
        "年龄",
        "圆锥角膜家族史",
        "文化程度",
        "Ambrósio’s relational thickness in the horizontal profile",
        "Dist. Apex-Thin.Loc. [mm](Dist. C-T)",
        "揉眼睛的频率",
        "每天使用电子屏幕（手机、电脑等）的总时间（小时）",
        "Pachy Apex(CCT)",
        "Pachy Prog Index Min.",
        "Steepest point of the front surface keratometry displacement in the x-axis",
        "每天在黑暗环境中使用电子屏幕的时间（小时）",
        "是否患有过敏性疾病？",
        "Mean eccentricity in the central 30 degrees by Fourier analysis",
        "感到工作/学习压力很大？",
        "睡觉时是否偏好把手或手臂垫放在眼睛上？",
        "Root-mean-square of total aberrations of whole cornea",
        "Maximum keratometry of the front surface",
        "Minimum Ambrósio relational thickness",
        "常在大量灰尘环境中工作或生活？",
        "阅读书籍",
        "Pachy Prog Index Avg.",
        "Km B (mm)",
        "K1 F (D)",
        "是否患有其他疾病？",
        "Average Ambrósio relational thickness",
        "惯用手",
        "职业",
        "幼年时居住地",
        "BAD Df",
        "Steepest point of the front surface keratometry displacement in the y-axis",
        "Pachy Min (TCT)",
        "RMS (CF)",
        "Index of surface variance",
        "BAD Dp",
        "常于夜间工作/学习？",
        "每天在户外阳光/紫外线下活动时间（小时）",
        "甲状腺疾病",
        "Biomechanically corrected intraocular pressure",
        "Pachy Prog Index Max.",
        "RMS LOA (CB)",
        "X position of the thinnest point",
        "Keratoconus index",
        "是否对某些物质过敏？",
        "春季角结膜炎",
        "是否用过外源性性激素药物？",
        "RMS HOA (CF)",
        "RMS (CB)",
        "RMS HOA (CB)",
        "Y position of the thinnest point",
        "Ratio between the central deformation and the average of the peripheral deformation determined at 1mm",
        "每次揉眼持续时间",
        "index of vertical asymmetry",
        "Intraocular pressure",
        "RMS LOA (CF)",
        "Corneal volume in a 3mm diameter zone around the corneal apex",
        "Integrated radius",

    ]
    df = df[top_k[:61]]  # include label
    # Top 40 keys
    # df = df[[
    #     "性别",
    #     "幼年时家庭经济状况",
    #     "BMI",
    #     "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？",
    #     "年龄",
    #     "圆锥角膜家族史",
    #     "文化程度",
    #     "Ambrósio’s relational thickness in the horizontal profile",
    #     "Dist. Apex-Thin.Loc. [mm](Dist. C-T)",
    #     "揉眼睛的频率",
    #     "每天使用电子屏幕（手机、电脑等）的总时间（小时）",
    #     "Pachy Prog Index Min.",
    #     "Steepest point of the front surface keratometry displacement in the x-axis",
    #     "Pachy Apex(CCT)",
    #     "每天在黑暗环境中使用电子屏幕的时间（小时）",
    #     "是否患有过敏性疾病？",
    #     "感到工作/学习压力很大？",
    #     "Mean eccentricity in the central 30 degrees by Fourier analysis",
    #     "睡觉时是否偏好把手或手臂垫放在眼睛上？",
    #     "Root-mean-square of total aberrations of whole cornea",
    #     "Maximum keratometry of the front surface",
    #     "Minimum Ambrósio relational thickness",
    #     "常在大量灰尘环境中工作或生活？",
    #     "阅读书籍",
    #     "Pachy Prog Index Avg.",
    #     "Km B (mm)",
    #     "K1 F (D)",
    #     "是否患有其他疾病？",
    #     "Average Ambrósio relational thickness",
    #     "惯用手",
    #     "职业",
    #     "BAD Df",
    #     "幼年时居住地",
    #     "Steepest point of the front surface keratometry displacement in the y-axis",
    #     "BAD Dp",
    #     "Pachy Min (TCT)",
    #     "RMS (CF)",
    #     "Index of surface variance",
    #     "常于夜间工作/学习？",
    #     "每天在户外阳光/紫外线下活动时间（小时）",
    #     "label"
    # ]]
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

    # top 20 keys
    # df = df[[
    #     "性别",
    #     "幼年时家庭经济状况",
    #     "BMI",
    #     "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？",
    #     "年龄",
    #     "圆锥角膜家族史",
    #     "文化程度",
    #     "Dist. Apex-Thin.Loc. [mm](Dist. C-T)",
    #     "Ambrósio’s relational thickness in the horizontal profile",
    #     "Pachy Prog Index Min.",
    #     "每天使用电子屏幕（手机、电脑等）的总时间（小时）",
    #     "揉眼睛的频率",
    #     "Steepest point of the front surface keratometry displacement in the x-axis",
    #     "Pachy Apex(CCT)",
    #     "每天在黑暗环境中使用电子屏幕的时间（小时）",
    #     "是否患有过敏性疾病？",
    #     "感到工作/学习压力很大？",
    #     "Mean eccentricity in the central 30 degrees by Fourier analysis",
    #     "睡觉时是否偏好把手或手臂垫放在眼睛上？",
    #     "Root-mean-square of total aberrations of whole cornea",
    #     "label"
    # ]]
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
    # if args.cls.lower() == "single":
    #     df['label'] = df[handle_columns].sum(axis=1).apply(lambda x: 1 if x != 0 else 0)
    # else:  # multi_class
    #     df['label'] = df[handle_columns].apply(lambda row: mode(row, nan_policy='omit').mode, axis=1)
    df['label'] = df[handle_columns].apply(lambda row: mode(row, nan_policy='omit').mode, axis=1)
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
    # select features
    return df

def preprocess_digit(args, df):  # digit to word
    def replace_with_description(x, threshold, discriminator):

        if discriminator == DSMTR_LT:  # <
            # if threshold < 0.1:
            #     bins = [threshold + threshold * 5, threshold + threshold * 10, threshold + threshold * 15]
            # elif threshold < 100:
            #     bins = [threshold + 5, threshold + 10, threshold + 15]
            # else:
            #     bins = [threshold + 50, threshold + 100, threshold + 200]
            # bins = threshold if threshold < 1 and isinstance(threshold, float) else [threshold-10,threshold,threshold+10]

            # return (
            #     f'{x} (less than {threshold} is normal, greater than or equal to {threshold} and less than {bins[0]} is mildly abnormal, '
            #     f'greater than or equal to {bins[0]} and less than {bins[1]} is moderately abnormal, '
            #     f'and greater than or equal to {bins[1]} is severely abnormal)')

            return (
                f'{x} (less than {threshold} is normal, greater than or equal to {threshold} is abnormal.'
                f'the larger the value, the more abnormal it tends to be.')
                # 'and the smaller the value, the more normal it tends to be.'
        else:
            # if threshold < 0.1:
            #     bins = [threshold - threshold * 5, threshold - threshold * 10, threshold - threshold * 15]
            # elif threshold < 100:
            #     bins = [threshold - 5, threshold - 10, threshold - 15]
            # else:
            #     bins = [threshold - 50, threshold - 100, threshold - 150]
            # # bins = threshold if threshold < 1 and isinstance(threshold, float) else [threshold-10,threshold,threshold+10]
            # return (
            #     f'{x} (greater than or equal to {threshold} is normal, less than {threshold} and greater than or equal to {bins[0]} is mildly abnormal, '
            #     f'less than {bins[0]} and greater than or equal to {bins[1]} is moderately abnormal, '
            #     f'and less than {bins[1]} is severely abnormal)')
            return (
                    f'{x} (greater than {threshold} is normal, less than or equal to {threshold} is abnormal.'
                    f'the smaller the value, the more abnormal it tends to be.')
            # 'and the larger the value, the more normal it tends to be.'

    # substitute digit to word
    for k, v in rule_yiduo.items():
        if (k not in df.columns.values.tolist()) or (not v): continue
        df[k] = df[k].apply(lambda x: replace_with_description(x, v['threshold'], v['discriminator']))

        # df[k] = pd.cut(df[k], bins=v["bins"], labels=v["labels"], right=False, include_lowest=False)

    return df

def preprocess_d2w(args, df): # digit to word
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

    if args.selected:
        sft_train_dataset_file_name = args.file_name['sft_fs_train_data']
        dpo_train_dataset_file_name = args.file_name['dpo_fs_train_data']
        test_dataset_file_name = args.file_name['test_fs_data']
        test_label_file_name = args.file_name['test_fs_label']
    else:
        sft_train_dataset_file_name = args.file_name['sft_train_data']
        dpo_train_dataset_file_name = args.file_name['dpo_train_data']
        test_dataset_file_name = args.file_name['test_data']
        test_label_file_name = args.file_name['test_label']

    with open(os.path.join(args.path['dataset_dir'], sft_train_dataset_file_name), 'w') as f:
        for i, row in enumerate(sft_train_queries):
            if i == 0: logger.info(f'============== sft train query: {row}')
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(sft_train_queries, f, ensure_ascii=False) # json
    with open(os.path.join(args.path['dataset_dir'], dpo_train_dataset_file_name), 'w') as f:
        for i, row in enumerate(dpo_train_queries):
            if i == 0: logger.info(f'============== dpo train query: {row}')
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(dpo_train_queries, f, ensure_ascii=False)
    with open(os.path.join(args.path['dataset_dir'], test_dataset_file_name), 'w') as f:
        for i, row in enumerate(test_queries):
            if i == 0: logger.info(f'============== test query : {row}')
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(test_queries, f, ensure_ascii=False)
    with open(os.path.join(args.path['dataset_dir'], test_label_file_name), 'w') as f:
        for i, row in enumerate(test_labels):
            if i == 0: logger.info(f'============== test label: {row}')
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        # json.dump(test_labels, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
