#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/8 12:07

from collections import defaultdict
from glob import glob

import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

from module.mix_metric import *

__all__ = ["ModelProperty"]


class ModelProperty(object):

    def __init__(self, f_score, info_list=None, d_optimize=None, cs_conf=None):
        self.f_score = f_score
        self.info_list = info_list
        self.d_optimize = d_optimize

        if type(cs_conf) == str:
            self.cs_conf = yaml.load(open(cs_conf), Loader=yaml.FullLoader)["arg_combine_score"]
        else:
            self.cs_conf = cs_conf

        self.score = self._score()
        self.last_select = pd.DataFrame()

    def _score(self):

        df_score = pd.read_csv(self.f_score, sep="\t")

        if self.info_list:
            rslt = []
            for name, file in self.info_list.items():
                df_t = pd.read_csv(file, sep="\t")
                df_t["Dataset"] = name
                if "ID" in df_t.columns:
                    df_t = df_t.rename(columns={"ID": "SampleID"})
                rslt.append(df_t)
            df_info = pd.concat(rslt, ignore_index=True, sort=False)
            df_score = pd.merge(df_info, df_score, on="SampleID", how="outer")

        if self.d_optimize:
            df_opt = pd.concat([pd.read_csv(f, sep="\t") for f in glob(f"{self.d_optimize}/Optimize*")])
            df_score = pd.merge(df_score, df_opt, on="SampleID", how="outer", suffixes=["", "_y"])

        df_score = df_score[~df_score.Score.isna()]
        return df_score

    def cutoff(self, spec, **kwargs):

        df_pred = self.select(**kwargs)
        df_pred = df_pred.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df_pred["Train_Group"] = df_pred.Train_Group.apply(lambda x: 0 if x == "Healthy" else 1)
        fpr, tpr, thresholds = roc_curve(df_pred["Train_Group"], df_pred["Score"], drop_intermediate=False)
        df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
        df_roc["tnr"] = 1 - df_roc.fpr

        # thresholds取最接近spec的，然后统计小于该cutoff下的最大Healthy score和紧接着上面一个Cancer score的均值最为新cutoff
        cutoff = df_roc.iloc[(df_roc.tnr - spec).abs().argsort()].iloc[0]["thresholds"]
        nearest = df_pred[(df_pred["Score"] < cutoff) & (df_pred.Train_Group == 0)]
        nearest = nearest["Score"].iloc[0]  # spec为0会报错
        nearest_up = df_pred[(df_pred["Score"] > nearest) & (df_pred.Train_Group == 1)]
        nearest_up = nearest_up["Score"].iloc[-1] if len(nearest_up) else cutoff
        nearest_up = nearest_up if nearest_up < cutoff else cutoff
        cutoff = np.mean([nearest, nearest_up])
        return cutoff

    def auc(self, **kwargs):
        """返回模型在各个数据集下面的auc结果"""

        df_pred = self.select(**kwargs)
        df_pred["Train_Group"] = df_pred.Train_Group.apply(lambda x: 0 if x == "Healthy" else 1)
        if len(df_pred["Train_Group"]) != 1:
            auc = roc_auc_score(df_pred["Train_Group"], df_pred["Score"])
        else:
            auc = 1
        return auc

    def performance(self, cutoff, **kwargs):
        """统计各个模型性能指标"""

        df_pred = self.select(**kwargs)

        df_pred.Train_Group = df_pred.Train_Group.apply(lambda x: 0 if x == "Healthy" else 1)
        tn = len(df_pred[(df_pred.Train_Group == 0) & (df_pred["Score"] < cutoff)])
        fp = len(df_pred[(df_pred.Train_Group == 0) & (df_pred["Score"] >= cutoff)])
        tp = len(df_pred[(df_pred.Train_Group == 1) & (df_pred["Score"] >= cutoff)])
        fn = len(df_pred[(df_pred.Train_Group == 1) & (df_pred["Score"] < cutoff)])
        accuracy = (tp + tn) / (tp + tn + fp + fn + 0.000000001)
        specificity = tn / (tn + fp + 0.000000001)
        sensitivity = tp / (tp + fn + 0.000000001)

        rslt = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy}
        return rslt

    def sensitivity(self, cutoff, **kwargs):
        """统计在不同的cutoff下的模型Sensitivity"""

        return self.performance(cutoff, **kwargs).get("sensitivity")

    def specificity(self, cutoff, **kwargs):
        """统计模型在不同cutoff下，对于不同数据集的specificity"""

        return self.performance(cutoff, **kwargs).get("specificity")

    def accuracy(self, cutoff, **kwargs):
        """统计模型在不同cutoff下，对于不同数据集的accuracy"""

        return self.performance(cutoff, **kwargs).get("accuracy")

    def pred_classify(self, cutoff, **kwargs):
        """样本预测分类以及正确性结果"""

        df_pred = self.select(**kwargs)

        df_pred["Score"] = df_pred["Score"].astype(float)
        df_pred["Train_Group"] = df_pred["Train_Group"].apply(lambda x: "Healthy" if x == "Healthy" else "Cancer")
        df_pred["Pred_Group"] = df_pred.apply(lambda x: "Cancer" if x["Score"] >= cutoff else "Healthy", axis=1)
        df_pred["Pred_Stat"] = df_pred.apply(lambda x: "Right" if x.Pred_Group == x.Train_Group else "Wrong", axis=1)
        return df_pred

    def rep_consistency(self, cutoff, **kwargs):

        df_pred = self.pred_classify(cutoff=cutoff, **kwargs)

        df_pred = df_pred.groupby(["OptimizeName", "SampleGroup"]).agg({
            "Pred_Stat": lambda x: sorted([len(x[x == "Right"]), len(x[x == "Wrong"])], reverse=True)[0],
            "SampleID": "size"
        }).reset_index()
        df_pred = df_pred[df_pred.SampleID > 1]
        value = df_pred.Pred_Stat.sum() / df_pred.SampleID.sum()
        return value

    @staticmethod
    def kolmogorov_smirnov(df_1, df_2):

        report = stats.ks_2samp(df_1["Score"], df_2["Score"], alternative="two-sided", mode="auto")
        value = 1 - report.statistic
        return value

    def sd(self, cutoff=None, **kwargs):
        df_pred = self.pred_classify(cutoff=cutoff, **kwargs)

        df_pred = df_pred.groupby(["OptimizeName", "SampleGroup"]).agg(
            {"SampleID": "size", "Score": "var"}).reset_index()
        df_pred = df_pred[df_pred.SampleID > 1]
        value = np.sqrt(np.sum(df_pred.Score)) / (len(df_pred) - 1)
        return value

    def combine_score(self, cutoff):
        """统计模型的combine score"""

        stat_value = {}
        for cs_args in self.cs_conf:

            dataset = cs_args.get("Dataset")
            opt = cs_args.get("Optimize")

            # 统计各项结果
            if cs_args["mode"] == "auc":
                value = self.auc(Dataset=dataset, OptimizeName=opt)

            elif cs_args["mode"] == "sensitivity":
                value = self.sensitivity(cutoff=cutoff, Dataset=dataset, OptimizeName=opt)

            elif cs_args["mode"] == "acc":
                value = self.accuracy(cutoff=cutoff, Dataset=dataset, OptimizeName=opt)

            elif cs_args["mode"] == "rep":
                value = self.rep_consistency(cutoff=cutoff, Dataset=dataset, OptimizeName=opt)

            elif cs_args["mode"] == "ks":
                df_1 = self.pred_classify(cutoff=cutoff, Dataset=cs_args.get("Dataset")[0])
                df_2 = self.pred_classify(cutoff=cutoff, Dataset=cs_args.get("Dataset")[1])
                value = self.kolmogorov_smirnov(df_1, df_2)

            elif cs_args["mode"] == "sd":
                value = 1 - self.sd(cutoff=cutoff, Dataset=dataset, OptimizeName=opt)
            else:
                raise ValueError("combinescore error")

            stat_value[cs_args["name"]] = value

        c_values = [stat_value[t["name"]] for t in self.cs_conf]
        c_weight = [t["weight"] for t in self.cs_conf]
        stat_value["CombineScore"] = combine_metrics(c_values, c_weight)

        return stat_value

    def select(self, drop_duplicate=True, exclude=None, **kwargs):
        """ 根据筛选条件，过滤样本信息

        字符型的列会筛选对应字段值，数值型的列会筛选对应范围。

        :return:
        """

        df_score = self.score.copy()

        for field, value in kwargs.items():
            if not value:
                continue
            if type(value) != list:
                df_score = df_score[df_score[field] == value]
            elif df_score.dtypes[field] == object:
                df_score = df_score[df_score[field].isin(value)]
            elif df_score.dtypes[field] in ["int64", "float64"]:
                df_score = df_score[(df_score[field] >= value[0]) & (df_score[field] <= value[1])]

        if exclude:
            for field, value in exclude.items():
                df_score = df_score[df_score[field] != value]

        df_score = df_score[~df_score.Score.isna()]
        if drop_duplicate:
            df_score = df_score.drop_duplicates(subset="SampleID", keep="last")

        self.last_select = df_score.copy()
        return df_score

    @property
    def summary(self):
        """结果一览

        Train_Group, Detatil_Group, Stage_TNM, Stage_E_L, Sex, TubeType, Lib, Project, OptimizeName

        """

        rslt = {"score": pd.DataFrame(), "summary": pd.DataFrame(), "combine_score": pd.DataFrame()}
        fields = ["Train_Group", "Detail_Group", "Stage_TNM", "Stage_E_L", "Sex", "TubeType",
                  "Lib", "Project", "Dataset"]
        if self.d_optimize:
            fields.append("OptimizeName")

        for field in fields:
            for field_value in set(self.score[field]):
                df_t = self.select(**{field: field_value})
                df_t.insert(0, "ScoreGroup1", field)
                df_t.insert(1, "ScoreGroup2", field_value)
                rslt["score"] = pd.concat([rslt["score"], df_t], ignore_index=True, sort=False)

        # summary
        tmp = []
        for spec in [0.90, 0.95, 0.98]:
            for field in fields:
                for field_value in set(self.score[field]):
                    try:
                        kwargs = {field: field_value}
                        cutoff = self.cutoff(spec=spec, Dataset="train")
                        auc = self.auc(**kwargs)
                        sens = self.sensitivity(cutoff=cutoff, **kwargs)
                        valid_spec = self.specificity(cutoff=cutoff, **kwargs)
                        acc = self.accuracy(cutoff=cutoff, **kwargs)
                        tmp.append({
                            "ScoreGroup1": field,
                            "ScoreGroup2": field_value,
                            "TrainSpec": spec,
                            "Cutoff": cutoff,
                            "AUC": auc,
                            "Sensitivity": sens,
                            "Specificity": valid_spec,
                            "Accuracy": acc,
                        })
                    except:
                        pass

        rslt["summary"] = pd.DataFrame(tmp)

        # combine
        tmp = []
        if self.cs_conf:
            for spec in [0.90, 0.95, 0.98]:
                cutoff = self.cutoff(spec=spec, Dataset="train")
                score = self.combine_score(cutoff=cutoff)
                score = dict({"TrainSpec": spec, "Cutoff": cutoff}, **score)
                tmp.append(score)

        rslt["combine_score"] = pd.DataFrame(tmp)

        return rslt


