#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/2 11:33
# @File     : pipe_h2o_train_by_dataset_split.py
# @Project  : gsml


""" 之前的经典模型训练流程。（train 数据集多seed拆分，判断模型性能）

主要流程：
    1. 将Train 数据集拆多次分成Train，Test数据集
    2. 对于同一数据集，不同特征分别进行多个算法的模型训练。
    3. 同一数据集下，相同算法进行stacked
    4. 同一数据集下，所有算法进行stacked
"""

from itertools import product
import os
import logging
import subprocess
from collections import defaultdict

import coloredlogs
import pandas as pd

from module.error import *
from module.submit_lsf import submit_lsf
from module.split_dataset import split_dataset
from stats.model_property import ModelProperty

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class PipeH2oTrainByDatasetSplit(object):
    def __init__(self, train_info: list, pred_info: list, feature: list, prefix, d_output, weights_column=None,
                 seed_list: list = None, ratio=0.6, nfolds=10, algorithms: list = None, fold_assignment="stratified",
                 threads=10, epochs=50, reproducible=True, step: list = None, stratify=None):
        self.train_info = train_info
        self.pred_info = pred_info
        self.feature = self._feature(feature)
        self.prefix = prefix
        self.d_output = self._outdir(d_output)
        self.weights_column = weights_column
        self.seed_list = seed_list or list(range(1, 21))
        self.ratio = ratio
        self.nfolds = nfolds
        self.algorithms = algorithms or ["glm", "gbm", "rf", "dl", "xgboost"]
        self.fold_assignment = fold_assignment
        self.threads = threads
        self.epochs = epochs
        self.reproducible = reproducible
        self.step = step or ["ds_copy", "ds_split", "base_train", "stacked_train", "stat", "plot"]
        self.stratify = stratify

        self.f_gsml = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../gsml")
        self.d_rawdata = self._outdir(f"{self.d_output}/RawData")
        self.d_dataset = self._outdir(f"{self.d_output}/Dataset")
        self.d_model = self._outdir(f"{self.d_output}/Model")
        self.d_stat = self._outdir(f"{self.d_output}/Stat")
        self.d_plot = self._outdir(f"{self.d_output}/Plot")
        self.d_log = self._outdir(f"{self.d_output}/Log")
        self.f_total_feature = f"{self.d_dataset}/Total.feature.csv"

        self.model_dict = {
                "glm": "Train_H2OGeneralizedLinear",
                "gbm": "Train_H2OGradientBoosting",
                "rf": "Train_H2ORandomForest",
                "df": "Train_H2oDeepLearning",
                "xgboost": "Train_H2OXGBoost",
                "dl": "Train_H2oDeepLearning",
            }

    def copy_raw_data(self):
        """保存原始数据"""

        for file in self.train_info:
            subprocess.check_output(f"cp {file} {self.d_rawdata}/{os.path.basename(file)}", shell=True)

        for pred in self.pred_info:
            file = pred.split(":")[-1]
            subprocess.check_output(f"cp {file} {self.d_rawdata}/{os.path.basename(file)}", shell=True)

        for k in self.feature.keys():
            for file in self.feature[k]:
                subprocess.check_output(f"cp {file} {self.d_rawdata}/{os.path.basename(file)}", shell=True)

    def split_dataset(self):
        """拆分数据集"""

        for seed in self.seed_list:
            f_train = f"{self.d_dataset}/{self.prefix}-Train-seed{seed}.info.list"
            f_test = f"{self.d_dataset}/{self.prefix}-TrainTest-seed{seed}.info.list"
            split_dataset(self.train_info, f_train=f_train, f_test=f_test, train_size=self.ratio, random_state=seed,
                          stratify=self.stratify)

        # 合并所有feature
        rslt = []
        for feature_list in self.feature.values():
            df_t = pd.concat([pd.read_csv(f) for f in feature_list], ignore_index=True, sort=False)
            rslt.append(df_t)

        df_feature = pd.DataFrame(columns=["SampleID"])
        for df_t in rslt:
            df_feature = pd.merge(df_feature, df_t, on="SampleID", how="outer", suffixes=["", "y"])
        df_feature.to_csv(self.f_total_feature, index=False)

    def train_base_model(self):
        """base model 训练"""

        cmd_list = []

        for seed, algorithm, feature in product(*[self.seed_list, self.algorithms, self.feature.keys()]):
            prefix = f"{self.prefix}-{feature}-{algorithm}-seed{seed}"
            d_output = f"{self.d_model}/seed{seed}/{algorithm}"
            pred_info = [f.split(':')[-1] for f in self.pred_info] + [f"{self.d_dataset}/{self.prefix}-TrainTest-seed{seed}.info.list"]
            train_info = f"{self.d_dataset}/{self.prefix}-Train-seed{seed}.info.list"

            cmd = f"{self.f_gsml} {self.model_dict.get(algorithm)} " \
                  f"--train_info {train_info}" \
                  f"{''.join([' --pred_info ' + f for f in pred_info])} " \
                  f"{''.join([' --feature ' + f for f in self.feature[feature]])} " \
                  f"--prefix {prefix} " \
                  f"--d_output {d_output} " \
                  f"--nfolds {self.nfolds} " \
                  f" {'--weights_column' if self.weights_column else ''} " \
                  f"--fold_assignment {self.fold_assignment} " \
                  f"--threads {self.threads}"

            if algorithm == "dl":
                cmd += f" --epochs {self.epochs} "
                cmd += f" {'--reproducible' if self.reproducible else ''}  "
            cmd_list.append((prefix, cmd))

        submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.threads, wait=1)

    def train_base_stacked(self):
        """算法内部stacked"""

        cmd_list = []
        for seed, algorithm in product(*[self.seed_list, self.algorithms]):
            d_output = f"{self.d_model}/seed{seed}/Stacked"
            prefix = f"{self.prefix}-Stacked{algorithm}-seed{seed}"
            pred_info = [f.split(':')[-1] for f in self.pred_info] + [f"{self.d_dataset}/{self.prefix}-TrainTest-seed{seed}.info.list"]
            train_info = f"{self.d_dataset}/{self.prefix}-Train-seed{seed}.info.list"

            cmd = f"{self.f_gsml} Train_H2OStackedEnsemble " \
                  f"--d_base_models {self.d_model}/seed{seed}/{algorithm} " \
                  f"--prefix {prefix} " \
                  f"-o {d_output} " \
                  f"--feature {self.f_total_feature} " \
                  f"--train_info {train_info}" \
                  f"{''.join([' --pred_info ' + f for f in pred_info])} "
            cmd_list.append((prefix, cmd))
        submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.threads, wait=1)

    def train_stacked_stacked(self):
        """所有模型stacked"""

        cmd_list = []
        for seed in self.seed_list:
            d_output = f"{self.d_model}/seed{seed}/StackedStacked"
            prefix = f"{self.prefix}-StackedStacked-seed{seed}"
            d_base_models = [f"{self.d_model}/seed{seed}/{a}" for a in self.algorithms]
            pred_info = [f.split(':')[-1] for f in self.pred_info] + [f"{self.d_dataset}/{self.prefix}-TrainTest-seed{seed}.info.list"]
            train_info = f"{self.d_dataset}/{self.prefix}-Train-seed{seed}.info.list"

            cmd = f"{self.f_gsml} Train_H2OStackedEnsemble " \
                  f"{''.join([' --d_base_models ' + d for d in d_base_models])} " \
                  f"--prefix {prefix} " \
                  f"-o {d_output} " \
                  f"--feature {self.f_total_feature} " \
                  f"--train_info {train_info}" \
                  f"{''.join([' --pred_info ' + f for f in pred_info])} "
            cmd_list.append((prefix, cmd))
        submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.threads, wait=1)

    def stat(self):
        """合并各个模型的统计结果"""

        # 遍历所有模型
        models_list = []
        for seed, algorithm, feature in product(*[self.seed_list, self.algorithms, self.feature.keys()]):
            prefix = f"{self.prefix}-{feature}-{algorithm}-seed{seed}"
            d_output = f"{self.d_model}/seed{seed}/{algorithm}"
            f_score = f"{d_output}/{prefix}.Predict.tsv"
            tmp = {
                "ModelID": prefix,
                "Algorithm": algorithm,
                "Feature": feature,
                "Seed": seed,
                "f_score": f_score,
                "f_auc": f"{self.d_stat}/{prefix}.ModelStat.AUC.tsv",
                "f_performance": f"{self.d_stat}/{prefix}.ModelStat.Performance.tsv",
                "f_auc_sub_group": f"{self.d_stat}/{prefix}.ModelStat.AucSubGroup.tsv",
                "f_performance_sub_group": f"{self.d_stat}/{prefix}.ModelStat.PerformanceSubGroup.tsv",
            }
            models_list.append(tmp)

        for seed, algorithm in product(*[self.seed_list, self.algorithms]):
            d_output = f"{self.d_model}/seed{seed}/Stacked"
            prefix = f"{self.prefix}-Stacked{algorithm}-seed{seed}"
            f_score = f"{d_output}/{prefix}.Predict.tsv"

            tmp = {
                "ModelID": prefix,
                "Algorithm": "Stacked",
                "Feature": "Stacked",
                "Seed": seed,
                "f_score": f_score,
                "f_auc": f"{self.d_stat}/{prefix}.ModelStat.AUC.tsv",
                "f_performance": f"{self.d_stat}/{prefix}.ModelStat.Performance.tsv",
                "f_auc_sub_group": f"{self.d_stat}/{prefix}.ModelStat.AucSubGroup.tsv",
                "f_performance_sub_group": f"{self.d_stat}/{prefix}.ModelStat.PerformanceSubGroup.tsv",
            }
            models_list.append(tmp)

        for seed in self.seed_list:
            d_output = f"{self.d_model}/seed{seed}/StackedStacked"
            prefix = f"{self.prefix}-StackedStacked-seed{seed}"
            f_score = f"{d_output}/{prefix}.Predict.tsv"

            tmp = {
                "ModelID": prefix,
                "Algorithm": "StackedStacked",
                "Feature": "StackedStacked",
                "Seed": seed,
                "f_score": f_score,
                "f_auc": f"{self.d_stat}/{prefix}.ModelStat.AUC.tsv",
                "f_performance": f"{self.d_stat}/{prefix}.ModelStat.Performance.tsv",
                "f_auc_sub_group": f"{self.d_stat}/{prefix}.ModelStat.AucSubGroup.tsv",
                "f_performance_sub_group": f"{self.d_stat}/{prefix}.ModelStat.PerformanceSubGroup.tsv",
            }
            models_list.append(tmp)

        # 每个模型分析性能
        info_list = defaultdict(dict)
        for seed in self.seed_list:
            info_list[seed]["Train"] = f"{self.d_dataset}/{self.prefix}-Train-seed{seed}.info.list"
            info_list[seed]["TrainTest"] = f"{self.d_dataset}/{self.prefix}-TrainTest-seed{seed}.info.list"
            info_list[seed] = dict(info_list[seed], **{f.split(":")[0]: f.split(":")[1] for f in self.pred_info})

        cmd_list = []
        for model in models_list:
            cmd = f"{self.f_gsml} ModelStat " \
                  f"--f_score {model['f_score']} " \
                  f"--d_output {self.d_stat} " \
                  f"--model_name {model['ModelID']} " \
                  f"{''.join([' --dataset ' + k + ',' + v for k, v in info_list[model['Seed']].items()])} " \
                  f"--skip_combine_score"
            cmd_list.append((f"ModelStat-{model['ModelID']}", cmd))
        submit_lsf(cmd_list, d_output=self.d_log, nthreads=4, wait=1)

        # 生成最终统计文件
        rslt = defaultdict(list)
        for model in models_list:
            df_auc = pd.read_csv(model['f_auc'], sep="\t")
            df_auc.insert(1, "Algorithm", model['Algorithm'])
            df_auc.insert(2, "Feature", model['Feature'])
            df_auc.insert(3, "Seed", model['Seed'])
            rslt['auc'].append(df_auc)

            df_performance = pd.read_csv(model['f_performance'], sep="\t")
            df_performance.insert(1, "Algorithm", model['Algorithm'])
            df_performance.insert(2, "Feature", model['Feature'])
            df_performance.insert(3, "Seed", model['Seed'])
            rslt['performance'].append(df_performance)

            df_auc_sub_group = pd.read_csv(model['f_auc_sub_group'], sep="\t")
            df_auc_sub_group.insert(1, "Algorithm", model['Algorithm'])
            df_auc_sub_group.insert(2, "Feature", model['Feature'])
            df_auc_sub_group.insert(3, "Seed", model['Seed'])
            rslt['auc_sub_group'].append(df_auc_sub_group)

            df_performance_sub_group = pd.read_csv(model['f_performance_sub_group'], sep="\t")
            df_performance_sub_group.insert(1, "Algorithm", model['Algorithm'])
            df_performance_sub_group.insert(2, "Feature", model['Feature'])
            df_performance_sub_group.insert(3, "Seed", model['Seed'])
            rslt['performance_sub_group'].append(df_performance_sub_group)

        df_auc = pd.concat(rslt['auc'], ignore_index=True, sort=False)
        df_auc.to_csv(f"{self.d_stat}/{self.prefix}.Total.ModelStat.AUC.tsv", sep="\t", index=False)

        df_performance = pd.concat(rslt['performance'], ignore_index=True, sort=False)
        df_performance.to_csv(f"{self.d_stat}/{self.prefix}.Total.ModelStat.Performance.tsv", sep="\t", index=False)

        df_auc_sub_group = pd.concat(rslt['auc_sub_group'], ignore_index=True, sort=False)
        df_auc_sub_group.to_csv(f"{self.d_stat}/{self.prefix}.Total.ModelStat.AucSubGroup.tsv", sep="\t", index=False)

        df_performance_sub_group = pd.concat(rslt['performance_sub_group'], ignore_index=True, sort=False)
        df_performance_sub_group.to_csv(f"{self.d_stat}/{self.prefix}.Total.PerformanceSubGroup.AUC.tsv", sep="\t", index=False)

    def run(self):
        """跑整个流程"""

        if "ds_copy" in self.step:
            logger.info(f"copy raw dataset")
            self.copy_raw_data()

        if "ds_split" in self.step:
            logger.info(f"split dataset and merge all features")
            self.split_dataset()

        if "base_train" in self.step:
            logger.info(f"train by each algorithms")
            self.train_base_model()

        if "stacked_train" in self.step:
            logger.info(f"train stacked model")
            self.train_base_stacked()
            self.train_stacked_stacked()

        if "stat" in self.step:
            logger.info(f"stat models")
            self.stat()

    @staticmethod
    def _feature(feature):
        """确定特征"""

        rslt = defaultdict(list)
        for f in feature:
            name, file = f.split(":")
            rslt[name].append(file)
        return rslt

    @staticmethod
    def _outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p
