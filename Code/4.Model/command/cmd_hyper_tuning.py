#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 14:40
# @Author  : shenny
# @File    : cmd_hyper_tuning.py
# @Software: PyCharm


import click

from module.hyper_tuning import HyperDANN


__all__ = ["cli_hyper_tuning"]


@click.group()
def cli_hyper_tuning():
    pass


@cli_hyper_tuning.command("HyperDANN")
@click.option("--num_cpus", default=1, type=click.INT, show_default=True, help="运行最大cpu")
@click.option("--num_gpus", default=-1, type=click.INT, show_default=True, help="运行最大gpu.-1就是有就用")
@click.option("--f_train", required=True, help="训练数据集")
@click.option("--f_valid", required=True, help="验证数据集")
@click.option("--f_feature", required=True, help="测试数据集")
@click.option("--f_hyper_params", required=True, help="超参数文件")
@click.option("--metric", default="valid__loss", show_default=True,
              help=f"监控指标.可以为valid__loss,valid__loss_class,valid__loss_domain,valid__accuracy,valid__recall,"
                   f"valid_f1等等")
@click.option("--mode", default="min", type=click.Choice(["min", "max"]), show_default=True, help="mode of hyper tuning")
@click.option("--num_samples", default=1000, type=click.INT, show_default=True, help="最大实验次数")
@click.option("--time_budget_s", default=36000, type=click.INT, show_default=True, help="最大实验时间")
@click.option("--log_to_file", default=False, type=click.BOOL, show_default=True, help="是否保存日志")
@click.option("--local_dir", required=True, help="保存实验结果的目录")
@click.option("--name", required=True, help="实验名称")
@click.option("--scale_method", default="minmax", type=click.Choice(["minmax"]), show_default=True, help="数据标准化方法")
@click.option("--na_strategy", default="mean", type=click.Choice(["mean"]), show_default=True, help="缺失值填充策略")
@click.option("--early_strategies", show_default=True, multiple=True,
              help="早停策略,可以多个。格式为<metric>,<patience>,<mode>,<threshold>. valid__loss,20,min,0.0001")
@click.option("--f_init_params", help="初始化参数文件,如果是模型继续训练，需要使用这个参数")
@click.option("--model_state_dict", help="模型参数文件,如果是模型继续训练，需要使用这个参数")
@click.option("--opt_state_dict", help="优化器参数文件,如果是模型继续训练，需要使用这个参数")
@click.option("--disable_dann", default=False, type=click.BOOL, show_default=True, help="禁用DANN中的域对抗模块")
def cmd_hyper_dann(num_cpus,num_gpus, **kwargs):
    """DANN模型超参数调优"""



    tun = HyperDANN(num_cpus=num_cpus, num_gpus=num_gpus)
    tun.run(**kwargs)
