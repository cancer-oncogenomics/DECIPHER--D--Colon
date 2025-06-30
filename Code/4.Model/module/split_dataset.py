# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/6/30 23:21
# @File     : split_dataset.py
# @Project  : gsml

"""拆分数据集"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def split_dataset(dataset, f_train, f_test, train_size=None, random_state=None, shuffle=True, stratify=None):
    """ 拆分数据集

    :param dataset: <list>  每个拆分子数据集的名字前缀
    :param f_train: <str>  数据集输出目录
    :param f_test: <str>  数据集输出目录
    :param train_size: <int>  样本拆分时的种子
    :param random_state: <list>  每个子数据集的拆分比例
    :param shuffle: <list>  待拆分的数据
    :param stratify: <list>  待拆分的数据
    :return: <dict>
    """

    # 获得每个数据集的样本数量
    df_dataset = pd.concat([pd.read_csv(f, sep="\t") for f in dataset], ignore_index=True, sort=False)
    col_y = df_dataset.columns[0]
    stratify = df_dataset[stratify] if stratify else None


    df_t, df_v, _, _ = train_test_split(df_dataset, df_dataset[col_y], train_size=train_size, stratify=stratify,
                                        random_state=random_state, shuffle=shuffle)

    # 数据集拆分
    df_t.to_csv(f_train, sep="\t", index=False)
    df_v.to_csv(f_test, sep="\t", index=False)

    return {"train": df_t, "valid": df_v}


if __name__ == '__main__':
    import os
    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "../demo/dataset")
    datasets = [f"{path}/Train.info.list", f"{path}/Valid1.info.list"]
    _rslt = split_dataset(datasets, output_train="/dssg/home/sheny/test/123.tsv", output_test="/dssg/home/sheny/test/456.tsv")
