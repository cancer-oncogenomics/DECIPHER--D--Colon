#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 21:01
# @File     : cmd_train.py
# @Project  : gsml


"""模型训练"""

import click

from module.train import train_h2o_base, train_h2o_stack, train_gs_stack
from estimators.dann import GsDANN
from module import cluster
from module.save_model import  save_model
from module.load_model import load_model
from module import log

__all__ = ["cli_train"]


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli_train():
    """Command line tool for model training"""

    pass


@cli_train.command("Train_H2oDeepLearning", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2oDeepLearning",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
@click.option("--epochs",
              type=click.INT,
              default=50,
              show_default=True,
              help="How many times the dataset should be iterated (streamed), can be fractional."
              )
@click.option("--reproducible",
              is_flag=True,
              default=True,
              show_default=True,
              help="How many times the dataset should be iterated (streamed), can be fractional."
              )
def cmd_h2o_deep_learning(threads, **kwargs):
    """used H2oDeepLearning algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2oDeepLearning", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OGradientBoosting", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OGradientBoosting",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_gbm(threads, **kwargs):
    """used H2OGradientBoosting algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OGradientBoosting", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OGeneralizedLinear", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OGeneralizedLinear",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_glm(threads, **kwargs):
    """used H2OGeneralizedLinear algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OGeneralizedLinear", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2ORandomForest", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2ORandomForest",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_rm(threads, **kwargs):
    """used H2ORandomForest algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2ORandomForest", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OXGBoost", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-w", "--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
def cmd_h2o_xgboost(threads, **kwargs):
    """used H2OXGBoost algorithm to model training"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_base(algorithm="H2OXGBoost", **kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_H2OStackedEnsemble", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=True,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-b", "--d_base_models",
              multiple=True,
              show_default=True,
              help="The directory of base models"
              )
@click.option("--model_list",
              multiple=True,
              show_default=True,
              help="The path of base models"
              )
@click.option("-n", "--metalearner_nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds"
              )
@click.option("-d", "--metalearner_fold_assignment",
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              default="stratified",
              show_default=True,
              help="fold_assignment"
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
@click.option("--metalearner_nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="metalearner_nfolds"
              )
@click.option("--seed",
              type=click.INT,
              default=10,
              show_default=True,
              help="seed"
              )
@click.option("--metalearner_algorithm",
              default="auto",
              show_default=True,
              help="metalearner_algorithm"
              )
def cmd_h2o_stacked(threads, **kwargs):
    """Train by H2OStackedEnsemble"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_h2o_stack(**kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_GsStacked", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("-r", "--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("-p", "--pred_info",
              required=False,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("-f", "--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("-a", "--prefix",
              required=True,
              default="H2OXGBoost",
              help="Prefix of output files"
              )
@click.option("-o", "--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("-b", "--d_base_models",
              multiple=True,
              show_default=True,
              help="The directory of base models"
              )
@click.option("--model_list",
              multiple=True,
              show_default=True,
              help="The path of base models"
              )
@click.option("-n", "--nfolds",
              type=click.INT,
              default=10,
              show_default=True,
              help="nfolds. The MEAN algorithm is invalid."
              )
@click.option("-t", "--threads",
              type=click.INT,
              default=4,
              show_default=True,
              help="nthreads"
              )
@click.option("--seed",
              type=click.INT,
              default=10,
              show_default=True,
              help="seed"
              )
@click.option("--metalearner_algorithm",
              default="mean",
              show_default=True,
              type=click.Choice(["mean", "glm"]),
              help="metalearner_algorithm"
              )
@click.option("--re_pred",
              is_flag=True,
              show_default=True,
              help="The base model repredicts each sample"
              )
@click.option("--too_class",
              show_default=True,
              help="Too categories,like 'Breast,Colorectal,Gastric,Liver,Lung'"
              )
def cmd_gs_stacked(threads, **kwargs):
    """Train by Gs Stacked"""

    cluster.init(nthreads=threads, max_mem_size=f"{threads * 4 * 1000}M")
    try:
        train_gs_stack(**kwargs)
    finally:
        cluster.close()


@cli_train.command("Train_DANN", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--f_model", help="gsml模型路径")
@click.option("--f_train", required=True, help="训练数据集")
@click.option("--f_valid", required=True, help="验证数据集")
@click.option("--f_feature", required=True, help="特征文件")
@click.option("--d_output", required=True, help="保存实验结果的目录")
@click.option("--model_name", required=True, help="模型名称")
@click.option("--early_strategies", show_default=True, multiple=True,
              help="早停策略,可以多个。格式为<metric>,<patience>,<mode>,<threshold>. valid__loss,20,min,0.0001")
@click.option("--disable_dann", default=False, type=click.BOOL, show_default=True, help="禁用DANN中的域对抗模块")
@click.option("--retrain", default=False, type=click.BOOL, show_default=True, help="是否重头训练")
@click.option("--lr", type=click.FLOAT, show_default=True, help="学习率")
@click.option("--lambda_domain", type=click.FLOAT, show_default=True, help="DANN的lambda参数")
@click.option("--weight_decay", type=click.FLOAT, show_default=True, help="权重衰减")
@click.option("--scale_method", default="minmax", type=click.Choice(["minmax"]), help="特征缩放方法")
@click.option("--na_strategy", default="mean", type=click.Choice(["mean"]), help="缺失值填充方法")
@click.option("--batch_size", type=click.INT, show_default=True, help="batch_size")
@click.option("--epochs", type=click.INT, show_default=True, help="epochs")
@click.option("--out1", type=click.INT, show_default=True, help="第一个卷积层的输出通道数")
@click.option("--conv1", type=click.INT, show_default=True, help="第一个卷积层的卷积核大小")
@click.option("--pool1", type=click.INT, show_default=True, help="第一个池化层的池化核大小")
@click.option("--drop1", type=click.FLOAT, show_default=True, help="第一个池化层的随机失活率")
@click.option("--out2", type=click.INT, show_default=True, help="第二个卷积层的输出通道数")
@click.option("--conv2", type=click.INT, show_default=True, help="第二个卷积层的卷积核大小")
@click.option("--pool2", type=click.INT, show_default=True, help="第二个池化层的池化核大小")
@click.option("--drop2", type=click.FLOAT, show_default=True, help="第二个池化层的随机失活率")
@click.option("--fc1", type=click.INT, show_default=True, help="第一个全连接层的输出大小")
@click.option("--fc2", type=click.INT, show_default=True, help="第二个全连接层的输出大小")
@click.option("--drop3", type=click.FLOAT, show_default=True, help="第二个全连接层的随机失活率")
def cmd_train_dann(f_train, f_valid, f_feature, d_output, model_name, early_strategies, retrain, lr, weight_decay,
                   scale_method, na_strategy, batch_size, epochs, out1, conv1, pool1, drop1, out2, conv2, pool2, drop2,
                   fc1, fc2, drop3, disable_dann, f_model, lambda_domain):
    """DANN模型训练"""

    init_params = {"out1": out1, "conv1": conv1, "pool1": pool1, "drop1": drop1, "out2": out2, "conv2": conv2,
                   "pool2": pool2, "drop2": drop2, "fc1": fc1, "fc2": fc2, "drop3": drop3}

    if f_model and not retrain:
        log.info(f"载入已有模型: {f_model}", 1)
        model = load_model(f_model)
    else:
        model = GsDANN()
        model.init_framer(f_feature=f_feature, f_train=f_train, scale_method=scale_method, na_strategy=na_strategy)

    model.train(
    	f_train=f_train,
    	f_valid=f_valid,
    	f_feature=f_feature,
        early_strategies=early_strategies,
    	d_output=d_output,
    	model_name=model_name,
    	init_params=init_params,
    	lr=lr,
    	weight_decay=weight_decay,
    	lambda_domain=lambda_domain,
    	batch_size=batch_size,
    	epochs=epochs,
        disable_dann=disable_dann,
        retrain=retrain
    )

    save_model(model, d_output, model_name)
