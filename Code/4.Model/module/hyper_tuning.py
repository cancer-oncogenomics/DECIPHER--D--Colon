#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/4 13:17
# @Author  : shenny
# @File    : hyper_tuning.py
# @Software: PyCharm

"""超参搜索"""

from functools import partial


import pandas as pd
import ray
from ray import air, tune
from ray.tune.schedulers import AsyncHyperBandScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import torch
from torch.utils.data import TensorDataset
import yaml

from estimators.dann import train_dann
from module.frame import TorchFrame


class HyperDANN(object):
    """ DANN超参搜索模块

    :param num_cpus: int, default=1, 可用的cpu总数
    :param num_gpus: int, default=1, 可用的gpu总数
    :param device: str, default=None, 设备名称
    """

    def __init__(self, num_cpus=1, num_gpus=-1):


        num_gpus = num_gpus if num_gpus != -1 else (1 if torch.cuda.is_available() else 0)
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    @staticmethod
    def get_hyper_params(file: str):
        """ 从yaml文件中获取超参搜索的参数，并转换成ray.tune的参数格式
        :param file: str, yaml文件路径
        """

        conf = yaml.load(open(file, "r"), Loader=yaml.FullLoader)
        params = {}
        for k, v in conf.items():
            if v["distribution"] == "uniform":
                params[k] = tune.uniform(v["value"][0], v["value"][1])
            elif v["distribution"] == "loguniform":
                params[k] = tune.loguniform(v["value"][0], v["value"][1])
            elif v["distribution"] == "quniform":
                params[k] = tune.quniform(v["value"][0], v["value"][1], v["step"])
            elif v["distribution"] == "choice":
                params[k] = tune.choice(v["value"])
            else:
                raise ValueError(f"distribution error. {v}")
        return params

    def partial_train(self, f_train: str, f_valid: str, f_feature: str, scale_method: str, na_strategy: str, **kwargs):
        """ 生成训练函数的偏函数
        :param f_train: 训练数据集。必须包含Response和Domain列
        :param f_valid: 验证数据集。必须包含Response和Domain列
        :param f_feature: 特征文件
        :param scale_method: 特征缩放方法。[minmax]
        :param na_strategy: 特征NA替换策略。[mean]
        :param kwargs: 其他参数。会都传递给train函数
        :return:
        """

        # 读取数据
        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)
        df_valid = pd.read_csv(f_valid, sep="\t", low_memory=False)
        df_train_domain = df_train[~df_train.Domain.isna()]
        df_valid_domain = df_valid[~df_valid.Domain.isna()]

        # 数据预处理
        framer = TorchFrame()
        framer.fit(df_feature, df_train, ["Response", "Domain"], scale_method, na_strategy)

        # 创建数据集
        ds_train = framer.create_tensor_dataset(df_feature, df_train, ["Response"])
        domain_train = framer.create_tensor_dataset(df_feature, df_train_domain, ["Response", "Domain"])
        ds_valid = framer.create_tensor_dataset(df_feature, df_valid, ["Response"])
        domain_valid = framer.create_tensor_dataset(df_feature, df_valid_domain, ["Response", "Domain"])

        # 生成训练函数的偏函数
        func = partial(self.train,
                       ds_train=ray.put(ds_train),
                       ds_valid=ray.put(ds_valid),
                       domain_valid=ray.put(domain_valid),
                       domain_train=ray.put(domain_train),
                       input_size=len(framer.features),
                       num_class=len(framer.classes["Response"]),
                       num_domain=len(framer.classes["Domain"]),
                       **kwargs
                       )
        return func

    @staticmethod
    def train(config, ds_train, ds_valid, domain_valid, domain_train, input_size: int, num_class: int,
              num_domain: int, f_init_params: str = None, model_state_dict: str = None, opt_state_dict: str = None,
              disable_dann: bool = True, early_strategies: str = None):
        """ 训练函数
        :param config: 超参
        :param ds_train: 训练数据集
        :param ds_valid: 验证数据集
        :param domain_valid: domain验证数据集
        :param domain_train: domain训练数据集
        :param input_size: 特征数
        :param num_class: 类别数
        :param num_domain: domain数
        :param f_init_params: 模型初始化参数。如果是对模型的继续训练，需要提供模型的初始化参数
        :param model_state_dict: 模型状态参数。如果是对模型的继续训练，需要提供模型的状态参数
        :param opt_state_dict: 优化器状态参数。如果是对模型的继续训练，需要提供优化器的状态参数
        :param disable_dann: 是否开启DANN模型的域对抗模块
        :param early_strategies: 提前停止策略。格式为"metric,patience,mode,threshold"
        """

        # 读取数据
        ds_train = ray.get(ds_train)
        ds_valid = ray.get(ds_valid)
        domain_valid = ray.get(domain_valid)
        domain_train = ray.get(domain_train)

        # 确定模型初始化参数
        params = ["out1", "conv1", "pool1", "drop1", "out2", "conv2", "pool2", "drop2", "fc1", "fc2", "drop3"]
        init_params = yaml.load(open(f_init_params, "r"), Loader=yaml.FullLoader) if f_init_params else config
        init_params = {k: v for k, v in init_params.items() if k in params}
        init_params["input_size"] = input_size
        init_params["num_class"] = num_class
        init_params["num_domain"] = num_domain

        # 训练模型
        train_dann(ds_train=ds_train,
                   ds_valid=ds_valid,
                   domain_train=domain_train,
                   domain_valid=domain_valid,
                   init_params=init_params,
                   lr=config["lr"],
                   weight_decay=config["weight_decay"],
                   batch_size=int(config["batch_size"]),
                   lambda_domain=config["lambda_domain"],
                   model_state_dict=model_state_dict,
                   opt_state_dict=opt_state_dict,
                   device="cuda" if torch.cuda.is_available() else "cpu",
                   epochs=int(config["epochs"]),
                   disable_dann=disable_dann,
                   early_strategies=early_strategies,
                   verbose=4
                   )

    def run(self, f_train: str, f_valid: str, f_feature: str, f_hyper_params: str, metric: str, mode: str,
            num_samples: int, time_budget_s: int, log_to_file: bool, local_dir: str, name: str,
            scale_method: str, na_strategy: str, early_strategies: list = None, f_init_params: str = None,
            model_state_dict: str = None, opt_state_dict: str = None, disable_dann: bool = True):
        """ 使用ray.tune进行超参搜索
        :param f_train: 训练数据集
        :param f_valid: 验证数据集
        :param f_feature: 特征文件
        :param f_hyper_params: 超参搜索空间文件
        :param metric: 要优化的指标。这个指标应该通过 tune.report() 报告。如果设置了，将传递给搜索算法和调度器。
        :param mode: 优化模式，必须是 ["min", "max"] 之一。决定是最小化还是最大化指标属性。如果设置了，将传递给搜索算法和调度器。
        :param num_samples:从超参数空间中采样的次数。默认为 1。如果提供了 grid_search，网格将重复 num_samples 次。
        :param time_budget_s: 全局时间预算（秒），超过后所有试验将停止。也可以是 datetime.timedelta 对象。
        :param log_to_file: 将 stdout 和 stderr 日志记录到 Tune 的试验目录中的文件。如果为 False（默认），不写入文件。
        :param local_dir:  本地目录路径。如果提供，将在此处写入试验结果。否则，将在 ~/ray_results 写入结果。
        :param name:  试验名称。如果提供，将在此处写入试验结果。否则，将在 ~/ray_results 写入结果。
        :param early_strategies: 提前停止策略。格式为"metric,patience,mode,threshold"
        :param f_init_params: 模型初始化参数。如果是对模型的继续训练，需要提供模型的初始化参数
        :param model_state_dict: 模型状态参数。如果是对模型的继续训练，需要提供模型的状态参数
        :param opt_state_dict: 优化器状态参数。如果是对模型的继续训练，需要提供优化器的状态参数
        :param disable_dann: 是否开启DANN模型的域对抗模块
        :param scale_method: 特征缩放方法。[minmax]
        :param na_strategy: 特征NA替换策略。[mean]
        """

        # 设置超参搜索算法
        search_algo = TuneBOHB(metric=metric, mode=mode)

        # 设置调度器
        scheduler = HyperBandForBOHB(metric=metric, mode=mode)

        # 确定超参搜索空间
        hyper_params = self.get_hyper_params(f_hyper_params)

        # 确定训练函数
        train_func = self.partial_train(f_train, f_valid, f_feature,
                                        f_init_params=f_init_params,
                                        model_state_dict=model_state_dict,
                                        opt_state_dict=opt_state_dict,
                                        disable_dann=disable_dann,
                                        early_strategies=early_strategies,
                                        scale_method=scale_method,
                                        na_strategy=na_strategy)

        # 启动超参搜索
        analysis = tune.run(
            train_func,
            config=hyper_params,
            search_alg=search_algo,
            scheduler=scheduler,
            fail_fast=False,
            raise_on_failed_trial=False,
            resources_per_trial={"gpu": 1, "cpu": 1} if torch.cuda.is_available() else {"cpu": 1},
            num_samples=num_samples,
            time_budget_s=time_budget_s,
            log_to_file=log_to_file,
            local_dir=local_dir,
            name=name,
            resume="AUTO",
            callbacks=[SaveBestModelCallback(metric=metric, mode=mode, filename=f"{local_dir}/best_model_performance.txt")]
        )

        # 获取最佳结果
        best_config = analysis.get_best_config(metric="valid__loss", mode="min")
        print(f"best_config: {best_config}")
        with open(f"{local_dir}/best_config.yaml", "w") as fw:
            fw.write(yaml.dump(best_config))

    def __del__(self):
        ray.shutdown()



class SaveBestModelCallback(tune.Callback):
    """保存最佳模型的回调函数"""

    def __init__(self, metric, mode, filename):
        self.metric = metric
        self.mode = mode
        self.best_performance = None
        self.filename = filename

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if self.mode == "max":
            is_better = self.best_performance is None or result[self.metric] > self.best_performance
        else:
            is_better = self.best_performance is None or result[self.metric] < self.best_performance

        if is_better:
            self.best_performance = result[self.metric]
            with open(self.filename, "w") as fw:
                col, data = [], []
                for k, v in result.items():
                    if k != "config":
                        col.append(k)
                        data.append(str(v))
                    else:
                        for k1, v1 in v.items():
                            col.append(k1)
                            data.append(str(v1))
                fw.write("\t".join(col) + "\n")
                fw.write("\t".join(data) + "\n")
