#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/2 8:47
# @File     : train.py
# @Project  : gsml

from glob import glob
import json
import logging
import os.path
import subprocess
import typing as t

import yaml
from pathlib import Path
import coloredlogs
import torch

from estimators.deeplearning import H2ODeepLearning
from estimators.gbm import H2OGradientBoosting
from estimators.glm import H2OGeneralizedLinear
from estimators.random_forset import H2ORandomForest
from estimators.xgboost import H2OXGBoost
from estimators.stackedensemble import H2OStackedEnsemble
from estimators.stacked_gs import GsStacked, GsStackedToo
from estimators.dann import DANN
from module import cluster
from module.frame import GsFrame
from module.save_model import save_model
from module.error import ArgsError
from module.load_model import load_model
from module.preprocess import MinMaxScale
from module.get_hyper_params import get_hyper_params
from module.submit_lsf import submit_lsf


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


__all__ = ["train_h2o_base", "train_h2o_stack"]


def train_h2o_base(algorithm, d_output=None, prefix=None, train_info=None, pred_info=None, feature=None,
                   weights_column=None, nfolds=10, fold_assignment="stratified", **kwargs):
    """ h2o单算法训练基本方法

    :param fold_assignment:
    :param nfolds:
    :param algorithm: 算法名称
    :param d_output: 模型输出路径
    :param prefix: 模型名字前缀
    :param train_info:  模型训练数据集
    :param pred_info:  模型预测数据集
    :param feature:  特征文件
    :param weights_column:  模型训练数据集文件中的权重列
    :param kwargs:  模型训练的其他参数
    :return:
    """

    algorithm_dict = {
        "H2oDeepLearning": H2ODeepLearning,
        "H2OGradientBoosting": H2OGradientBoosting,
        "H2OGeneralizedLinear": H2OGeneralizedLinear,
        "H2ORandomForest": H2ORandomForest,
        "H2OXGBoost": H2OXGBoost,
    }
    if algorithm not in algorithm_dict.keys():
        msg = f"algorithm must in {algorithm_dict.keys()}: <{algorithm}>"
        logger.error(msg)
        raise ArgsError(msg)

    # 生成数据集
    logger.info(f"generate GsFrame...")
    gf_train = GsFrame(dataset_list=train_info, feature_list=feature)
    gf_pred = GsFrame(dataset_list=pred_info, feature_list=feature) if pred_info else None

    # 模型训练
    logger.info(f"{algorithm} training...")
    model = algorithm_dict.get(algorithm)(nfolds=nfolds,
                                          fold_assignment=fold_assignment,
                                          **kwargs)
    model.train(x=gf_train.c_features,
                y="Response",
                training_frame=gf_train,
                predict_frame=gf_pred,
                weights_column=weights_column
                )

    # 模型预测
    if pred_info:
        model.predict(predict_frame=gf_pred)

    save_model(model=model, path=d_output, prefix=prefix)

    logger.info(f"success!")


def train_h2o_stack(base_models=None, model_list=None, d_base_models=None, d_output=None, prefix="H2oStackedEnsemble", threads=10,
                    train_info=None, pred_info=None, feature=None, metalearner_algorithm="auto", metalearner_nfolds=10,
                    metalearner_fold_assignment="stratified", **kwargs
                    ):
    """ h2o stacked 模型训练

    :param base_models:
    :param d_base_models:
    :param d_output:
    :param prefix:
    :param threads:
    :param train_info:
    :param pred_info:
    :param feature:
    :param metalearner_algorithm:
    :param metalearner_nfolds:
    :param metalearner_fold_assignment:
    :param kwargs:
    :return:
    """

    # 初始化h2o server
    logger.info(f"connect h2o server. <nthreads: {threads}; max_mem_size: {threads * 4 * 1000}M>")
    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")

    # 生成数据集
    logger.info(f"generate GsFrame...")
    gf_train = GsFrame(dataset_list=train_info, feature_list=feature)
    gf_pred = GsFrame(dataset_list=pred_info, feature_list=feature) if pred_info else None

    # 载入base_model
    logger.info(f"load base models...")
    base_models = base_models or []
    if not base_models and d_base_models:
        for f_mode in [m for d in d_base_models for m in glob(f"{d}/*gsml")]:
            model = load_model(f_mode, use_predict=True)
            base_models.append(model.model.model_id)
    elif not base_models and model_list:
        for f_mode in model_list:
            model = load_model(f_mode, use_predict=True)
            base_models.append(model.model.model_id)

    # 模型训练
    logger.info(f"H2OStackedEnsemble training...")
    model = H2OStackedEnsemble(metalearner_algorithm=metalearner_algorithm,
                               metalearner_nfolds=metalearner_nfolds,
                               metalearner_fold_assignment=metalearner_fold_assignment,
                               base_models=base_models,
                               **kwargs
                               )

    model.train(x=gf_train.c_features,
                y="Response",
                training_frame=gf_train,
                predict_frame=gf_pred,
                )

    save_model(model=model, path=d_output, prefix=prefix)
    cluster.close()
    logger.info(f"success!")


def train_gs_stack(base_models=None, model_list=None, d_base_models=None, d_output=None, prefix="Gs--Stacked",
                   train_info=None, pred_info=None, feature=None, metalearner_algorithm="mean",
                   nfolds=10, seed=-1, re_pred=False, too_class=None):

    """自研的stacked模型算法"""

    # 生成数据集
    logger.info(f"generate GsFrame...")
    gf_train = GsFrame(dataset_list=train_info, feature_list=feature)
    gf_pred = GsFrame(dataset_list=pred_info, feature_list=feature) if pred_info else None

    logger.info(f"load base models...")
    base_models = base_models or []
    if not base_models and d_base_models:
        for index, f_mode in enumerate([m for d in d_base_models for m in glob(f"{d}/*gsml")]):
            model_id = os.path.basename(f_mode).replace(".gsml", "")
            model = load_model(f_mode, use_predict=True)
            base_models.append( (f"{model_id}.{index}", model))
    elif not base_models and model_list:
        for index, f_mode in enumerate(model_list):
            model_id = os.path.basename(f_mode).replace(".gsml", "")
            model = load_model(f_mode, use_predict=True)
            base_models.append( (f"{model_id}.{index}", model) )

    # 模型训练
    if not too_class:
        logger.info(f"gs stack training...")
        model = GsStacked(base_models=base_models, d_output=d_output, model_id=prefix,
                          metalearner_algorithm=metalearner_algorithm)
        model.train(x=gf_train.c_features,
                    y="Response",
                    training_frame=gf_train,
                    predict_frame=gf_pred,
                    nfolds=nfolds,
                    seed=seed,
                    re_pred=re_pred,
                    )

        save_model(model=model, path=d_output, prefix=prefix)
        logger.info(f"success!")

    else:
        logger.info(f"gs stack training by TOO...")
        model = GsStackedToo(base_models=base_models, d_output=d_output, model_id=prefix,
                             metalearner_algorithm=metalearner_algorithm, too_class=too_class)
        model.train(x=gf_train.c_features,
                    y="Response",
                    training_frame=gf_train,
                    predict_frame=gf_pred,
                    nfolds=nfolds,
                    seed=seed,
                    re_pred=re_pred,
                    )

        save_model(model=model, path=d_output, prefix=prefix)
        logger.info(f"success!")

