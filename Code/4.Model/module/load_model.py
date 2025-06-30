#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/4 5:14

import os

import h2o
import joblib
import torch


__all__ = ["load_model"]


def load_model(f_model, use_predict=False):
    """ 载入一个模型

    :param use_predict:
    :param f_model:
    :return:
    """

    # 载入实例
    model_name = os.path.basename(f_model).replace(".gsml", "")
    model = joblib.load(f_model)

    model.d_output = os.path.dirname(f_model)
    try:
        model.d_base_models = os.path.join(model.d_output, f"BaseModel.{model_name}")
    except:
        pass

    # 载入模型
    if use_predict:
        if model.algorithm.startswith("H2o"):
            model.model = h2o.load_model(f"{f_model.replace('.gsml', '.model')}")
    return model
