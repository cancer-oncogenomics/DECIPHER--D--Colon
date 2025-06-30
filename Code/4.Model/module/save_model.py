#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/13 13:38

import os

import h2o
import torch
import joblib

__all__ = ["save_model"]


def save_model(model, path, prefix, skip_h2o=False):
    """ 保存一个模型实例

    :param prefix: 输出文件前缀
    :param path: 输出文件路径
    :param model: 模型实例
    :param skip_h2o: 不保存h2o的模型实例
    :return:
    """

    model.d_output = path

    # 保存模型
    if model.algorithm.startswith("H2o") and not skip_h2o:
        h2o.save_model(model.model, path=path, export_cross_validation_predictions=True, force=True, filename=f"{prefix}.model")

    # 保存模型权重
    try:
        model.varimp.to_csv(f"{path}/{prefix}.VarImp.tsv", sep="\t", index=False)
    except:
        pass

    # 保存基本类
    model.model = None
    joblib.dump(model, filename=f"{path}/{prefix}.gsml")

    # 保存得分
    model._score.to_csv(f"{path}/{prefix}.Predict.tsv", sep="\t", index=False)
