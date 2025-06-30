#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/9/27 13:24
# @File     : tabulate.py
# @Project  : gsml


"""生成频数表"""


import numpy as np
import pandas as pd


def tabulate(file, stat_cols: list=None, stat_types: list=None, header_col="", digits=1, f_output=None):
    """ 频数表统计

    :param file: info信息表
    :param stat_cols: 待统计的列
    :param stat_types: 待统计的列的类型
    :param header_col: 表头列
    :param digits: 小数保留位数
    :param f_output: 输出结果文件
    """

    df_info = pd.read_csv(file, sep="\t", na_values=["-"], dtype={c: t for c, t in zip(stat_cols, stat_types)})

    # 空值替换
    for c, t in zip(stat_cols, stat_types):
        df_info[c] = df_info[c].fillna("NA" if t == "str" else np.nan)

    # 确定要统计的列
    col_names = list(set(df_info[header_col]))
    # 确定要统计的行.形如“原始列名||统计信息”
    row_names = []
    for c, t in zip(stat_cols, stat_types):
        row_names.append(f"{c}||{c}||head")
        if t == "str":
            names = [f"{c}||{i}||{t}" for i in list(set(df_info[c]))]
        elif t == "float":
            names = [f"{c}||{i}||{t}" for i in ["Mean(SD)", "Median[Min,Max]"]]
        else:
            names = []
        row_names.extend(names)

    # 逐行统计信息
    rslt = []
    for row in row_names:
        raw_col, value, t = row.split("||")
        if t == "head":
            tmp = dict({"RowName": value}, **{c: "-" for c in col_names})
            rslt.append(tmp)
        if t == "str":
            tmp = {"RowName": value}
            for col_name in col_names:
                df_t = df_info[df_info[header_col] == col_name]
                total = df_t.shape[0]
                count = df_t[df_t[raw_col] == value].shape[0]
                pct = round(count / total * 100, digits)
                cell_value = f"{count}({pct}%)"
                tmp[col_name] = cell_value
            rslt.append(tmp)
        elif t == "float":
            tmp = {"RowName": value}
            for col_name in col_names:
                df_t = df_info[df_info[header_col] == col_name]
                describe = dict(df_t[raw_col].describe().round(digits))
                if value == "Mean(SD)":
                    cell_value = f"{describe['mean']}({describe['std']})"
                    tmp[col_name] = cell_value
                elif value == "Median[Min,Max]":
                    cell_value = f"{describe['50%']}[{describe['min']},{describe['max']}]"
                    tmp[col_name] = cell_value
            rslt.append(tmp)

    df_stat = pd.DataFrame(rslt).set_index("RowName")
    if f_output:
        df_stat.to_csv(f_output, sep="\t")

    return df_stat

if __name__ == '__main__':

    tabulate(
        f"/dssg02/InternalResearch02/sheny/Mercury/2022-09-10_Dataset0831/Analyze/Ds0831-test1/Dataset/info/Ds0831-test1.all.info.list",
        stat_cols=["Sex", "Age", "StageTnm"],
        stat_types=["str", "float", "str"],
        digits=1,
        header_col="GroupLevel2",
        f_output="/dssg/home/shen/test/123.tsv"
    )