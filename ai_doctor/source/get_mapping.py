import pandas as pd
import json


def excel_to_json(file_path, sheet_name, key_col, value_col, output_json_path):
    """
    将Excel中的两列转换为JSON格式并保存。

    :param file_path: Excel文件路径
    :param sheet_name: 工作表名称
    :param key_col: 用作JSON键的列名
    :param value_col: 用作JSON值的列名
    :param output_json_path: 输出JSON文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(file_path, sheet_name=0)

    # 检查是否存在指定的列
    if key_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns {key_col} and {value_col} must exist in the Excel sheet")

    # 转换为字典
    data_dict = dict(zip(df[key_col], df[value_col]))

    # 保存为JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)

    print(f"JSON file saved to {output_json_path}")


# 示例使用
file_path = 'abbr_mapping.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
key_col = 'Abbreviation'  # 替换为用于键的列名
value_col = 'Full Term'  # 替换为用于值的列名
output_json_path = 'output.json'  # 替换为你想保存的JSON文件路径

excel_to_json(file_path, sheet_name, key_col, value_col, output_json_path)
