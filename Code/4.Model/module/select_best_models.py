#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/21 14:16
# @File     : select_best_models.py
# @Project  : gsml


"""根据不同的指标， 筛选最佳的模型"""

import os
import pandas as pd

from module.load_model import load_model


class SelectBestModels(object):
    """ 根据不同的指标，选择最佳的base model

    :arg d_output: 结果输出路径
    :arg prefix: 结果输出路径
    :arg base_models: base model的路径
    :arg leaderboard_frame: 结果输出路径
    :arg select_method: 结果输出路径
    :arg f_pred: 结果输出路径


    """

    def __init__(self, d_output, prefix, leaderboard_frame, base_models: list = None, select_method="mean",
                 f_pred=None):
        self.d_output = self.outdir(d_output)
        self.prefix = prefix
        self.base_models = base_models
        self.leaderboard_frame = leaderboard_frame
        self.select_method = select_method
        self.f_pred = f_pred

        self.df_pred = pd.read_csv(f_pred, sep="\t") if f_pred else pd.DataFrame()




    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p
