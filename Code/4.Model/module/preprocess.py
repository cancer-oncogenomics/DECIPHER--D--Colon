#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 9:34
# @Author  : shenny
# @File    : preprocess.py
# @Software: PyCharm

"""特征预处理"""

import logging

import coloredlogs
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class MinMaxScale(object):
    """最大最小值缩放

    :param X: 需要缩放的数组
    :param na_strategy: 缺失值填充策略
    """

    def __init__(self, X: np.ndarray, na_strategy="mean"):

        self.imputer = SimpleImputer(strategy=na_strategy)  # 缺失值填充实例
        self.scaler = preprocessing.MinMaxScaler()  # 数据缩放实例

        self.imputer.fit(X)
        self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """特征缩放"""

        data = self.imputer.transform(X)
        data = self.scaler.transform(data)
        return data
