#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 14:21
# @Author  : shenny
# @File    : get_hyper_params.py
# @Software: PyCharm

"""生成超参数字典"""

from itertools import product


def get_hyper_params(**hyper_params):
    """遍历超参数字典
    超参数字典形如: {"batch_size": '32,64', "num_epochs": '100,200'， 'lr': '0.01'}
    """

    # 计算所有参数的笛卡尔积
    hyper_params = {k: v if type(v) == list else [v] for k, v in hyper_params.items()}
    cartesian_product = product(*hyper_params.values())

    # 将笛卡尔积转换成字典
    hyper_params_dict = [dict(zip(hyper_params.keys(), item)) for item in cartesian_product]

    return hyper_params_dict
