#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/10/10 9:33
# @File     : pipe_combine_best_model.py
# @Project  : gsml


"""根据生成的各个特征的base model，排列组合，得到最优的stacked model"""

# import sys
# sys.path.insert(0, "/dssg/home/sheny/MyProject/gsml")

from itertools import combinations
import logging
import os

import coloredlogs
import pandas as pd
from glob import glob
from joblib import Parallel, delayed

from module.submit_lsf import submit_lsf
from module.error import *

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class PipeCombineBestModel(object):
    """根据生成的各个特征的base model，排列组合，得到最优的stacked model

    :param d_model_list: base model 路径 [(list), (cnv,/dssg/home/test/cnv)]

    """

    def __init__(self, d_model_list, d_output, train_info, pred_info: list=None, threads=10, n_top_models="2,3,4,5",
                 stacked_algo: list = None, feature_list: list = None, stat_cols=None):
        self.d_model_list = [i.split(",") for i in d_model_list]
        self.d_output = self.outdir(d_output)
        self.train_info = train_info.split(",")
        self.pred_info = [i.split(",") for i in pred_info]
        self.threads = threads
        self.n_top_models = [int(i) for i in n_top_models.split(",")]
        self.stacked_algo = stacked_algo or ["mean", "glm"]
        self.feature_list = feature_list
        self.stat_cols = stat_cols

        self.features = [i[0] for i in self.d_model_list]
        self.gsml = "/dssg/NGSPipeline/Mercury/gsml/gsml"

        # 相关目录
        self.d_log = self.outdir(f"{d_output}/log")
        self.d_stacked = self.outdir(f"{d_output}/Stacked")

    def select(self):

        # 确定模型基本信息
        logger.info("Determine the model and statistical results")
        rslt = []
        for feature, d_model in self.d_model_list:

            if len(glob(f"{d_model}/*.gsml")) == 0:
                raise SampleNotFound(f"Can not find any gsml file in {d_model}")

            for f_model in glob(f"{d_model}/*.gsml"):

                # 排除AllModels模型
                if "StackedEnsemble_AllModels" in f_model:
                    continue

                model_id = os.path.basename(f_model).rsplit(".", 1)[0]
                tmp = {
                    "Feature": feature,
                    "ModelID": model_id,
                    "f_model": f_model,
                    "AUC": f"{d_model}/{model_id}.ModelStat.AUC.tsv",
                    "AucSubGroup": f"{d_model}/{model_id}.ModelStat.AucSubGroup.tsv",
                    "CombineScore": f"{d_model}/{model_id}.ModelStat.CombineScore.tsv",
                    "PerformanceSubGroup": f"{d_model}/{model_id}.ModelStat.PerformanceSubGroup.tsv",
                    "Performance": f"{d_model}/{model_id}.ModelStat.Performance.tsv",
                }
                rslt.append(tmp)
            df_info = pd.DataFrame(rslt)
            df_info.to_csv(f"{self.d_output}/base_model.info.tsv", sep="\t", index=False)

        # 确定模型统计结果
        df_info = pd.read_csv(f"{self.d_output}/base_model.info.tsv", sep="\t")
        rslt = Parallel(n_jobs=self.threads)(delayed(self.get_train_auc)(s) for _, s in df_info.iterrows())
        df_stat = pd.DataFrame(rslt)
        df_stat = df_stat.sort_values(by="TrainAUC", ascending=False)
        df_stat.to_csv(f"{self.d_output}/base_model.stat.tsv", sep="\t", index=False)

        # 确定组合方式，
        cb_list = []
        for n in range(3, len(self.features) + 1):
            for this_features in combinations(self.features, n):
                for n_model in self.n_top_models:
                    for algo in self.stacked_algo:
                        tmp = {"features": this_features, "n_model": n_model, "algo": algo}
                        cb_list.append(tmp)

        # 生成stacked命令
        df_stack = pd.DataFrame([self.choose_and_stack(**cb, df_stat=df_stat) for cb in cb_list])
        df_stack.to_csv(f"{self.d_output}/base_model.command.tsv", sep="\t", index=False)
        commands = [(s.ID, s.command) for _, s in df_stack.iterrows()]
        submit_lsf(commands=commands, d_output=self.d_log, nthreads=6, wait=0.1, force=False)

        # 生成模型统计命令
        commands = []
        for _, s in df_stack.iterrows():
            cmd = f"{self.gsml} ModelStat " \
                  f"--f_score {self.d_stacked}/{s.ID}.Predict.tsv " \
                  f"--d_output {self.d_stacked} " \
                  f"--model_name {s.ID} " \
                  f"--skip_combine_score " \
                  f"--dataset {','.join(self.train_info)} "
            if self.pred_info:
                for n, f in self.pred_info:
                    cmd += f" --dataset {n},{f} "
            if self.stat_cols:
                cmd += f"  --stat_cols {self.stat_cols} "
            commands.append((f"stat--{s.ID}", cmd))
        submit_lsf(commands=commands, d_output=self.d_log, nthreads=4, wait=0.1, force=False)
        # submit_lsf(commands=commands, d_output=self.d_log, nthreads=4, wait=0.5, force=True)

        # 合并最终统计结果
        rslt = Parallel(n_jobs=self.threads)(delayed(self.stat_stacked_simple)(s) for _, s in df_stack.iterrows())
        df_all = pd.concat(rslt, ignore_index=True, sort=False)
        df_all.to_csv(f"{self.d_output}/stacked_model.stat.simple.tsv", sep="\t", index=False)

    def choose_and_stack(self, df_stat, features, n_model, algo):

        prefix = '--'.join(features) + '--top' + str(n_model) + '--' + algo
        # 挑选出每个特征的最优top模型
        model_list = []
        for n_feature in features:
            df_t = df_stat[df_stat.Feature == n_feature]
            top_models = list(df_t.iloc[0: n_model]["f_model"]) if len(df_t) >= n_model else list(df_t["f_model"])
            model_list.extend(top_models)

        cmd = f"{self.gsml} Train_GsStacked " \
              f"--train_info {self.train_info[-1]} " \
              f"--metalearner_algorithm {algo} " \
              f"--d_output {self.d_stacked} " \
              f"--prefix {prefix} "
        if self.pred_info:
            for _, f_info in self.pred_info:
                cmd += f" --pred_info {f_info} "
        for feature in self.feature_list:
            cmd += f" --feature {feature} "
        for model in model_list:
            cmd += f" --model_list {model} "

        rslt = {"ID": prefix, "features": ",".join(features), "top": n_model, "algo": algo, "command": cmd}

        return rslt

    @staticmethod
    def get_train_auc(s):
        df_auc = pd.read_csv(s.AUC, sep="\t")
        auc = df_auc.loc[df_auc.Group2 == "Train", "AUC"].iloc[0]
        rslt = {"Feature": s.Feature, "ModelID": s.ModelID, "TrainAUC": float(auc), "f_model": s.f_model}
        return rslt

    def stat_stacked_all(self, s):

        df_a1 = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.AUC.tsv", sep="\t")
        df_a2 = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.AucSubGroup.tsv", sep="\t")
        df_auc = pd.concat([df_a1, df_a2], ignore_index=True, sort=False)

        df_p1 = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.Performance.tsv", sep="\t")
        df_p2 = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.PerformanceSubGroup.tsv", sep="\t")
        df_per = pd.concat([df_p1, df_p2], ignore_index=True, sort=False)

        df_all = pd.merge(df_auc, df_per, on=["ModelID", "Spec", "CutoffDataset", "Cutoff", "Group1", "Group2", "Dataset"], how="outer")
        return df_all

    def stat_stacked_simple(self, s):

        try:
            df_auc = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.AUC.tsv", sep="\t")

            df_per = pd.read_csv(f"{self.d_stacked}/{s.ID}.ModelStat.Performance.tsv", sep="\t")

            df_all = pd.merge(df_auc, df_per, on=["ModelID", "Spec", "CutoffDataset", "Cutoff", "Group1", "Group2"], how="outer")
            return df_all
        except:
            pass

    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p


if __name__ == '__main__':
    PipeCombineBestModel(
        d_model_list=[
            "cnv,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/cnv",
            "combine2,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/combine2",
            "combine3,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/combine3",
            "frag,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/frag",
            "frag.arm,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/frag_arm",
            "griffin,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/griffin.854TF",
            "neomer,/dssg02/InternalResearch02/sheny/Mercury/2022-09-17_Ds0831Moodel/Analyze/Ds0831/BaseModel/neomer.genegroup",
        ],
        d_output="/dssg/home/sheny/test/ModelSelect",
        train_info="Train,/dssg02/InternalResearch02/Mercury/MercuryDataSet/PanCancer/PanCancer_2022-08-31/info/Ds0831.Train.info.list",
        pred_info=["Valid,/dssg02/InternalResearch02/Mercury/MercuryDataSet/PanCancer/PanCancer_2022-08-31/info/Ds0831.Valid.info.list"],
        feature_list=["/dssg02/InternalResearch02/Mercury/MercuryDataSet/PanCancer/PanCancer_2022-08-31/Features/Ds0831.AllFeature.csv"]
    ).select()


