#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/29 22:11
# @File     : pipe_h2o_automl_too_two_layer.py
# @Project  : gsml


"""两层automl too分析"""

from collections import defaultdict
from functools import reduce
from itertools import combinations
import logging
import os
import subprocess
from glob import glob

import coloredlogs
import pandas as pd
from joblib import Parallel, delayed

from model.model_base import GsModelStat
from module.submit_lsf import submit_lsf


__all__ = ["PipeH2oAutoMlTooTwoLayer"]


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

class PipeH2oAutoMlTooTwoLayer(object):

    def __init__(self, d_output, train_info, valid_info, features, cancer_list, nfold=5, nfold_seed=1, force=False,
                 nthreads=10, max_models=200):
        self.d_output = self.outdir(d_output)
        self.train_info = train_info
        self.valid_info = valid_info
        self.features = features
        self.nfold = nfold
        self.nfold_seed = nfold_seed
        self.force = force
        self.cancer_list = cancer_list
        self.nthreads = nthreads
        self.max_models = max_models

        self.f_gsml = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../gsml")

        self.d_model = self.outdir(f"{d_output}/Model")
        self.d_layer1 = self.outdir(f"{self.d_model}/layer1")
        self.d_layer2 = self.outdir(f"{self.d_model}/layer2")
        self.d_layer3 = self.outdir(f"{self.d_model}/layer3")
        self.d_log = self.outdir(f"{d_output}/Log")
        self.d_rslt = self.outdir(f"{d_output}/Result")

    def iter_step(self, step):
        """遍历不同步骤的一些目录"""

        rslt = defaultdict(list)

        # layer1 one cancer
        for cancer in self.cancer_list:
            for n_feature, f_feature in self.features.items():
                d_output = self.outdir(f"{self.d_layer1}/OneCancer/{cancer}/{n_feature}")
                tmp = {
                    "layer": 1,
                    "cancer_count": 1,
                    "cancer": cancer,
                    "n_feature": n_feature,
                    "f_feature": f_feature,
                    "train_info": f"{d_output}/train.info.list",
                    "valid_info": f"{d_output}/valid.info.list",
                    "d_output": d_output,
                }
                rslt["layer1_one_cancer"].append(tmp)

        # layer1 two cancer
        for cancer_list in combinations(self.cancer_list, 2):
            c_name = '-'.join(cancer_list)
            for n_feature, f_feature in self.features.items():
                d_output = self.outdir(f"{self.d_layer1}/TwoCancer/{c_name}/{n_feature}")
                tmp = {
                    "layer": 1,
                    "cancer_count": 2,
                    "cancer": c_name,
                    "n_feature": n_feature,
                    "f_feature": f_feature,
                    "train_info": f"{d_output}/train.info.list",
                    "valid_info": f"{d_output}/valid.info.list",
                    "d_output": d_output,
                }
                rslt["layer1_two_cancer"].append(tmp)

        # layer1
        rslt["layer1"] = rslt["layer1_one_cancer"] + rslt["layer1_two_cancer"]

        # layer2 one cancer
        for cancer in self.cancer_list:
            d_output = self.outdir(f"{self.d_layer2}/OneCancer/{cancer}")
            tmp = {
                "layer": 2,
                "cancer_count": 1,
                "cancer": cancer,
                "n_feature": "layer1_oneCancer_score",
                "f_feature": f"{d_output}/layer1_oneCancer_score.csv",
                "train_info": f"{d_output}/train.info.list",
                "valid_info": f"{d_output}/valid.info.list",
                "d_output": d_output,
            }
            rslt["layer2_one_cancer"].append(tmp)

        # layer2 two cancer
        for cancer_list in combinations(self.cancer_list, 2):
            c_name = '-'.join(cancer_list)
            d_output = self.outdir(f"{self.d_layer2}/TwoCancer/{c_name}")
            tmp = {
                "layer": 2,
                "cancer_count": 2,
                "cancer": c_name,
                "n_feature": "layer1_twoCancer_score",
                "f_feature": f"{d_output}/layer1_twoCancer_score.csv",
                "train_info": f"{d_output}/train.info.list",
                "valid_info": f"{d_output}/valid.info.list",
                "d_output": d_output,
            }
            rslt["layer2_two_cancer"].append(tmp)

        # layer2
        rslt["layer2"] = rslt["layer2_one_cancer"] + rslt["layer2_two_cancer"]

        # layer2 two cancer stacked
        for cancer_list in combinations(self.cancer_list, 2):
            c_name = '-'.join(cancer_list)
            d_output = self.outdir(f"{self.d_layer2}/TwoCancer/{c_name}")
            d_stacked_output = self.outdir(f"{self.d_layer2}/TwoCancer/{c_name}/Stacked")
            tmp = {
                "layer": 2,
                "cancer_count": 2,
                "cancer": c_name,
                "n_feature": "layer1_twoCancer_score",
                "f_feature": f"{d_output}/layer1_twoCancer_score.csv",
                "train_info": f"{d_output}/train.info.list",
                "valid_info": f"{d_output}/valid.info.list",
                "d_output": d_stacked_output,
                "model_id": f"{c_name}_stacked",
                "f_score": f"{d_stacked_output}/{c_name}_stacked.Predict.tsv"
            }
            rslt["layer2_two_cancer_stacked"].append(tmp)

        return rslt[step]


    def cmd_automl(self, cancer=None, n_feature=None, f_feature=None, d_output=None, include_algos=None, **kwargs):

        cancer = cancer.split("-")

        if not os.path.exists(f"{d_output}/{n_feature}.csv") or not os.path.exists(f"{d_output}/valid.info.list"):

            df_train = pd.read_csv(self.train_info, sep="\t")
            df_valid = pd.read_csv(self.valid_info, sep="\t")

            for c in cancer:
                if not len(df_train[df_train.Response == c]) or not len(df_valid[df_valid.Response == c]):
                    logger.error(f"{c} not in train/valid info")
                    raise ValueError(f"{c} not in train/valid info")

            df_train.Response = df_train.Response.apply(lambda x: "Cancer" if x in cancer else "Healthy")
            df_train.to_csv(f"{d_output}/train.info.list", sep="\t", index=False)
            df_valid.Response = df_valid.Response.apply(lambda x: "Cancer" if x in cancer else "Healthy")
            df_valid.to_csv(f"{d_output}/valid.info.list", sep="\t", index=False)
            all_ids = list(df_train.SampleID) + list(df_valid.SampleID)

            df_feature = pd.read_csv(f_feature)
            df_feature = df_feature[df_feature.SampleID.isin(all_ids)]
            df_feature.to_csv(f"{d_output}/{n_feature}.csv", index=False)

        cmd = f"{self.f_gsml} H2oAutoML " \
              f"--d_output {d_output}/BaseModel " \
              f"--prefix layer1_automl " \
              f"--feature {d_output}/{n_feature}.csv " \
              f"--train_info {d_output}/train.info.list " \
              f"--pred_info {d_output}/valid.info.list " \
              f"--nthreads {self.nthreads} " \
              f"--max_models {self.max_models} " \
              f"--nfolds {self.nfold} " \
              f"--seed {self.nfold_seed} " \
              f"--balance_classes " \
              f"--stopping_metric mean_per_class_error"
        if include_algos:
            cmd += f" --include_algos {include_algos}"
        return cmd

    def cmd_model_stacked(self, base_models=None, f_feature=None, train_info=None, valid_info=None, d_output=None,
                          model_id=None, **kwargs):
        d_basemodel = self.outdir(f"{d_output}/BaseModel")
        for f_model in base_models:
            try:
                subprocess.check_output(f"ln -s {f_model.replace('.gsml', '*')} {d_basemodel}", shell=True)
            except:
                pass

        cmd = f"{self.f_gsml} Train_H2OStackedEnsemble " \
              f"--train_info {train_info} " \
              f"--pred_info {valid_info} " \
              f"--feature {f_feature} " \
              f"--prefix {model_id} " \
              f"--d_output {d_output} " \
              f"--d_base_models {d_basemodel} " \
              f"--threads {self.nthreads} " \
              f"--metalearner_nfolds {self.nfold} " \
              f"--seed {self.nfold_seed} " \
              f"--metalearner_algorithm glm"
        return cmd

    @staticmethod
    def model_stat(model_id, f_score, train_info, valid_info, **kwargs):

        dataset = {"Train": train_info, "Valid": valid_info}
        ms = GsModelStat(f_score=f_score, dataset=dataset)
        # auc = ms.auc(Dataset="Valid")
        return {"model_id": model_id, "valid_auc": ms.auc(Dataset="Valid"), "train_auc": ms.auc(Dataset="Train")}

    @staticmethod
    def model_score(model_id,  f_score, **kwargs):
        df_score = pd.read_csv(f_score, sep="\t")
        df_score = df_score[["SampleID", "Score"]]
        df_score = df_score.rename(columns={"Score": model_id})
        return df_score

    def model_accuracy(self, f_score):
        df_pred = pd.read_csv(f_score, sep="\t")
        df_info = pd.read_csv(self.valid_info, sep="\t")
        df_pred = df_pred[df_pred.SampleID.isin(df_info.SampleID)]
        df_pred["predict"] = df_pred[self.cancer_list].idxmax(axis=1)
        df_pred = pd.merge(df_info[["SampleID", "Response"]], df_pred[["SampleID", "predict"]], on="SampleID", how="inner")
        acc = len(df_pred[df_pred.Response == df_pred.predict]) / len(df_pred)
        return acc


    def train(self):

        # layer1 train
        cmd_list = []
        logger.info(f"layer1 train")
        for step in self.iter_step("layer1"):
            cmd = self.cmd_automl(**step)
            cmd_list.append((f"layer1_train-{step['cancer']}-{step['n_feature']}", cmd))
        submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.nthreads + 2, wait=1, force=self.force)
        #
        # # 得到第一层所有basemodel的信息
        # logger.info(f"first layer, stat model info")
        # rslt = []
        # for step in self.iter_step("layer1"):
        #     for f_model in glob(f"{step['d_output']}/BaseModel/*.gsml"):
        #         model_name = os.path.basename(f_model).replace('.gsml', '')
        #         tmp = {"f_model": f_model,
        #                "f_score": f_model.replace(".gsml", ".Predict.tsv"),
        #                "model_id": f"{step['cancer']}-{step['n_feature']}-{model_name}"
        #                }
        #         tmp = dict(step, **tmp)
        #         rslt.append(tmp)
        # df_info = pd.DataFrame(rslt)
        # df_info.to_csv(f"{self.d_rslt}/layer1.model.info.tsv", sep="\t", index=False)
        #
        # # 统计auc
        # logger.info(f"first layer, stat model auc")
        # rslt = Parallel(n_jobs=self.nthreads)(delayed(self.model_stat)(**s) for _, s in df_info.iterrows())
        # df_auc = pd.DataFrame(rslt)
        # df_auc = pd.merge(df_info, df_auc, on="model_id", how="inner")
        # df_auc.to_csv(f"{self.d_rslt}/layer1.model.stat.tsv", sep="\t", index=False)
        #
        # # 统计得分
        # logger.info(f"first layer, stat model score")
        # rslt = Parallel(n_jobs=self.nthreads)(delayed(self.model_score)(**s) for _, s in df_info.iterrows())
        # df_score = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="inner"), rslt)
        # df_score.to_csv(f"{self.d_rslt}/layer1.model.score.tsv", sep="\t", index=False)
        #
        # # layer2 生成feature (auc top5)
        # df_score = pd.read_csv(f"{self.d_rslt}/layer1.model.score.tsv", sep="\t")
        # df_auc = pd.read_csv(f"{self.d_rslt}/layer1.model.stat.tsv", sep="\t")
        # logger.info(f"layer2, make feature")
        # best_ids = defaultdict(list)
        # for (_, cancer, _), df_group in df_auc.groupby(["cancer_count", "cancer", "n_feature"]):
        #     df_t = df_group.sort_values(by="train_auc", ascending=False).iloc[0: 5]
        #     best_ids[cancer].extend(df_t.model_id)
        # for step in self.iter_step("layer2"):
        #     df_step_score = df_score[["SampleID"] + best_ids[step['cancer']]]
        #     df_step_score.to_csv(f"{step['f_feature']}", index=False)
        #
        # # layer2 模型训练
        # cmd_list = []
        # logger.info(f"layer2 train")
        # for step in self.iter_step("layer2"):
        #     cmd = self.cmd_automl(include_algos="GLM,XGBoost,DRF,DeepLearning", **step)
        #     cmd_list.append((f"layer2_train-{step['cancer']}", cmd))
        # submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.nthreads + 2, wait=1, force=self.force)
        #
        # # 得到layer2所有basemodel的信息
        # logger.info(f"layer2, stat model info")
        # rslt = []
        # for step in self.iter_step("layer2"):
        #     for f_model in glob(f"{step['d_output']}/BaseModel/*.gsml"):
        #         model_name = os.path.basename(f_model).replace('.gsml', '')
        #         tmp = {"f_model": f_model,
        #                "f_score": f_model.replace(".gsml", ".Predict.tsv"),
        #                "model_id": f"{step['cancer']}-{model_name}"
        #                }
        #         tmp = dict(step, **tmp)
        #         rslt.append(tmp)
        # df_info = pd.DataFrame(rslt)
        # df_info.to_csv(f"{self.d_rslt}/layer2.model.info.tsv", sep="\t", index=False)
        #
        # # 统计auc
        # logger.info(f"layer2, stat model auc")
        # rslt = Parallel(n_jobs=self.nthreads)(delayed(self.model_stat)(**s) for _, s in df_info.iterrows())
        # df_auc = pd.DataFrame(rslt)
        # df_auc = pd.merge(df_info, df_auc, on="model_id", how="inner")
        # df_auc.to_csv(f"{self.d_rslt}/layer2.model.stat.tsv", sep="\t", index=False)
        #
        # # 获得每个癌种的最优基模型
        # logger.info(f"layer2, find beds models")
        # best_ids = defaultdict(list)
        # for (_, cancer, _), df_group in df_auc.groupby(["cancer_count", "cancer", "n_feature"]):
        #     df_t = df_group.sort_values(by="train_auc", ascending=False).iloc[0: 5]
        #     best_ids[cancer].extend(df_t.model_id)
        #
        # # layer2 model stacked
        # cmd_list = []
        # for step in self.iter_step("layer2_two_cancer_stacked"):
        #     model_ids = best_ids[step["cancer"]]
        #     model_list = list(df_info.loc[df_info.model_id.isin(model_ids), "f_model"])
        #     cmd = self.cmd_model_stacked(base_models=model_list, **step)
        #     cmd_list.append((f"layer2_stacked-{step['cancer']}", cmd))
        # submit_lsf(commands=cmd_list, d_output=self.d_log, nthreads=self.nthreads + 2, wait=1, force=self.force)
        #
        # # 统计得分
        # logger.info(f"layer2, stat model score")
        # rslt = Parallel(n_jobs=self.nthreads)(delayed(self.model_score)(**s) for _, s in df_info.iterrows())
        # df_score = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="inner"), rslt)
        #
        # for step in self.iter_step("layer2_two_cancer_stacked"):
        #     df_t = self.model_score(**step)
        #     df_score = pd.merge(df_score, df_t, on="SampleID", how="inner")
        # df_score.to_csv(f"{self.d_rslt}/layer2.model.score.tsv", sep="\t", index=False)
        #
        # # 获得layer3 的feature
        # logger.info(f"layer3, train feature")
        # rslt = []
        # for step in self.iter_step("layer2_one_cancer"):
        #     id_list = best_ids[step['cancer']]
        #     df_step_score = df_score[["SampleID"] + id_list].copy()
        #     df_step_score[f"{step['cancer']}_mean"] = df_step_score[id_list].mean(axis=1)
        #     rslt.append(df_step_score)
        # for step in self.iter_step("layer2_two_cancer_stacked"):
        #     df_step_score = df_score[["SampleID", step["model_id"]]].copy()
        #     rslt.append(df_step_score)
        # df_feature = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="inner"), rslt)
        # df_feature.to_csv(f"{self.d_layer3}/layer2.best.score.csv", index=False)
        #
        # # layer3 模型训练
        # cmd = f"{self.f_gsml} H2oAutoML " \
        #       f"--d_output {self.d_layer3}/BaseModel " \
        #       f"--prefix layer3_automl " \
        #       f"--feature {self.d_layer3}/layer2.best.score.csv " \
        #       f"--train_info {self.train_info} " \
        #       f"--pred_info {self.valid_info} " \
        #       f"--nthreads {self.nthreads} " \
        #       f"--max_models {self.max_models} " \
        #       f"--nfolds {self.nfold} " \
        #       f"--seed {self.nfold_seed} " \
        #       f"--balance_classes " \
        #       f"--stopping_metric mean_per_class_error " \
        #       f"--include_algos XGBoost,DRF,DeepLearning"
        # submit_lsf(commands=[("layer3_train", cmd)], d_output=self.d_log, nthreads=self.nthreads + 2, wait=1, force=self.force)
        #
        # # 得到layer3所有basemodel的信息
        # logger.info(f"layer3, stat model info")
        # rslt = []
        # for f_model in glob(f"{self.d_layer3}/BaseModel/*.gsml"):
        #         tmp = {"f_model": f_model,
        #                "f_score": f_model.replace(".gsml", ".Predict.tsv"),
        #                "model_id": os.path.basename(f_model).replace(".gsml", "")
        #                }
        #         rslt.append(tmp)
        # df_info = pd.DataFrame(rslt)
        #
        # # layer3 统计accuracy
        # df_info["accuracy"] = df_info.f_score.apply(self.model_accuracy)
        # df_info = df_info.sort_values(by="accuracy", ascending=False)
        # df_info.to_csv(f"{self.d_rslt}/layer3.model.stat.tsv", sep="\t", index=False)

        logger.info("done")

    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p