#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/1 18:08

from itertools import product
import os

import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from module.mix_metric import *


__all__ = ["GsModelStat"]


class GsModelStat(object):
    """ 机器学习模型的基本类。用于返回模型的一些基本属性
    """

    def __init__(self, f_score="", dataset=None, optimize=None, cs_conf=None, **kwargs):
        self.dataset = dataset or {}
        self.optimize = optimize or {}
        self.cs_conf = cs_conf

        self._score = pd.read_csv(f_score, sep="\t") if os.path.exists(f_score) else pd.DataFrame()
        self._df_ss = self._sample_sheet()

        self.last_select = pd.DataFrame()

    def set_dataset(self, dataset=None, optimize=None, cs_conf=None):
        self.dataset = dataset
        self.optimize = optimize or {}
        self.cs_conf = cs_conf
        self._df_ss = self._sample_sheet()

    def _sample_sheet(self):
        df_ss = pd.DataFrame()
        for name, file in self.dataset.items():
            df_t = pd.read_csv(file, sep="\t")
            df_t["Dataset"] = name
            df_ss = pd.concat([df_ss, df_t], ignore_index=True, sort=False)

        df_opt = pd.DataFrame(columns=["SampleID"])
        for name, file in self.optimize.items():
            df_t = pd.read_csv(file, sep="\t")
            df_opt = pd.concat([df_opt, df_t], ignore_index=True, sort=False)

        if len(df_ss) or len(df_opt):
            df_rslt = pd.merge(df_ss, df_opt, on="SampleID", how="left", suffixes=["", "_y"])
        else:
            df_rslt = pd.DataFrame()

        return df_rslt

    @property
    def score(self):
        df_score = pd.merge(self._df_ss, self._score, on="SampleID", how="outer", suffixes=["_x", ""])
        # if len(df_score[df_score.Score.isna()]):
        #     no_scores = set(list(df_score[df_score.Score.isna()]["SampleID"]))
        #     raise ValueError(f"Some Sample did not score. <{','.join(no_scores)}>")
        return df_score

    def cutoff(self, spec=None, sens=None, **kwargs):

        df_pred = self.select(**kwargs)
        df_pred = df_pred.sort_values(by="Score", ascending=False).reset_index(drop=True)
        df_pred["Response"] = df_pred.Response.apply(lambda x: 0 if x == "Healthy" else 1)
        fpr, tpr, thresholds = roc_curve(df_pred["Response"], df_pred["Score"], drop_intermediate=False)
        df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
        df_roc["tnr"] = 1 - df_roc.fpr

        # thresholds取最接近spec的，然后统计小于该cutoff下的最大Healthy score和紧接着上面一个Cancer score的均值最为新cutoff
        if spec:
            cutoff = df_roc.iloc[(df_roc.tnr - spec).abs().argsort()].iloc[0]["thresholds"]
            nearest = df_pred[(df_pred["Score"] < cutoff) & (df_pred.Response == 0)]
            if nearest.shape[0]:
                nearest = nearest["Score"].iloc[0]  # spec为0会报错
                nearest_up = df_pred[(df_pred["Score"] > nearest) & (df_pred.Response == 1)]
                nearest_up = nearest_up["Score"].iloc[-1] if len(nearest_up) else cutoff
                nearest_up = nearest_up if nearest_up < cutoff else cutoff
                cutoff = np.mean([nearest, nearest_up])
            return cutoff
        elif sens:
            cutoff = df_roc.iloc[(df_roc.tpr - sens).abs().argsort()].iloc[0]["thresholds"]
            return cutoff

    def auc(self, **kwargs):
        """返回模型在各个数据集下面的auc结果"""

        try:
            df_pred = self.select(**kwargs)
            df_pred["Response"] = df_pred.Response.apply(lambda x: 0 if x == "Healthy" else 1)
            auc = roc_auc_score(df_pred["Response"], df_pred["Score"])
        except Exception as error:
            print(error)
            auc = np.NaN

        return auc

    def performance(self, cutoff, **kwargs):
        """统计各个模型性能指标"""

        try:
            df_pred = self.select(**kwargs)
            df_pred.Response = df_pred.Response.apply(lambda x: 0 if x == "Healthy" else 1)
            tn = len(df_pred[(df_pred.Response == 0) & (df_pred["Score"] < cutoff)])
            fp = len(df_pred[(df_pred.Response == 0) & (df_pred["Score"] >= cutoff)])
            tp = len(df_pred[(df_pred.Response == 1) & (df_pred["Score"] >= cutoff)])
            fn = len(df_pred[(df_pred.Response == 1) & (df_pred["Score"] < cutoff)])
            accuracy = (tp + tn) / (tp + tn + fp + fn + 0.000000001)
            specificity = tn / (tn + fp + 0.000000001)
            sensitivity = tp / (tp + fn + 0.000000001)

            rslt = {"sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy,
                    "TP": tp, "FP": fp, "TN": tn, "FN": fn}
        except:
            rslt = {"sensitivity": np.NaN, "specificity": np.NaN, "accuracy": np.NaN, "TP": np.NaN,
                    "FP": np.NaN, "TN": np.NaN, "FN": np.NaN}
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
        df_pred["Response"] = df_pred["Response"].apply(lambda x: "Healthy" if x == "Healthy" else "Cancer")
        df_pred["Pred_Group"] = df_pred.apply(lambda x: "Cancer" if x["Score"] >= cutoff else "Healthy", axis=1)
        df_pred["Pred_Stat"] = df_pred.apply(lambda x: "Right" if x.Pred_Group == x.Response else "Wrong", axis=1)
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

    def summary(self, cutoff=None, skip_auc=False, skip_performance=False, skip_combine_score=False,
                skip_by_subgroup=False, stat_cols=None, **kwargs):
        """ 统计指定cutoff下，该模型的各个统计结果

        :param stat_cols:
        :param skip_by_subgroup:
        :param skip_combine_score:
        :param skip_performance:
        :param skip_auc:
        :param cutoff: cutoff分值
        :param stat_auc: 是否统计auc
        :param stat_performance: 是否统计performance
        :param stat_combine_score: 是否统计combine_score
        :return:
        """

        rslt = {}

        # 预测分类结果
        df_classify = self.pred_classify(cutoff=cutoff)
        df_classify = df_classify[["SampleID", "Score", "Response", "Pred_Group", "Pred_Stat"]]
        for k, v in kwargs.items():
            df_classify[k] = v
        rslt["classify"] = df_classify

        # 不同数据集和优化项目的auc结果
        rslt_auc = []
        if not skip_auc:
            for dataset in self.dataset.keys():
                value = {"Group1": "Dataset", "Group2": dataset, "AUC": self.auc(Dataset=dataset)}
                rslt_auc.append(dict(kwargs, **value))
            for optimize in self.optimize.keys():
                value = {"Group1": "Optimize", "Group2": optimize, "AUC": self.auc(OptimizeName=optimize)}
                rslt_auc.append(dict(kwargs, **value))
        df_auc = pd.DataFrame(rslt_auc).fillna("-")
        rslt["Auc"] = df_auc

        # 不同数据集和优化项目的performance结果
        rslt_performance = []
        if not skip_performance:
            for dataset in self.dataset.keys():
                value = {"Group1": "Dataset", "Group2": dataset}
                value.update(self.performance(cutoff=cutoff, Dataset=dataset))
                rslt_performance.append(dict(kwargs, **value))
            for optimize in self.optimize.keys():
                value = {"Group1": "Optimize", "Group2": optimize}
                value.update(self.performance(cutoff=cutoff, OptimizeName=optimize))
                rslt_performance.append(dict(kwargs, **value))
        df_performance = pd.DataFrame(rslt_performance).fillna("-")
        rslt["Performance"] = df_performance

        # 不同数据集和优化项目的combine score结果
        rslt_cs = []
        if not skip_combine_score:
            value = dict(kwargs, **self.combine_score(cutoff=cutoff))
            rslt_cs.append(value)
        df_cs = pd.DataFrame(rslt_cs)
        rslt["CombineScore"] = df_cs

        # 不同小组的auc结果
        rslt_auc = []
        if not skip_by_subgroup and stat_cols and not skip_auc:
            dataset_list = list(self.dataset.keys()) + [None]  # 会分别统计不同数据集下和所有样本下，各个项目的性能
            for dataset, col_name in product(dataset_list, stat_cols):
                for col_value in self._df_ss[col_name].dropna().unique():
                    value = {
                        "Dataset": dataset or "-",
                        "Group1": col_name,
                        "Group2": col_value,
                        "AUC": self.auc(Dataset=dataset, **{col_name: col_value})
                    }
                    if len(self.last_select):
                        rslt_auc.append(dict(kwargs, **value))
        df_sub_auc = pd.DataFrame(rslt_auc).fillna("-")
        rslt["AucSubGroup"] = df_sub_auc

        # 不同小组的performance结果
        rslt_performance = []
        if not skip_by_subgroup and stat_cols and not skip_performance:
            dataset_list = list(self.dataset.keys()) + [None]
            for dataset, col_name in product(dataset_list, stat_cols):
                for col_value in self._df_ss[col_name].dropna().unique():
                    value = {"Dataset": dataset or "-", "Group1": col_name, "Group2": col_value}
                    value.update(self.performance(cutoff=cutoff, Dataset=dataset, **{col_name: col_value}))
                    if len(self.last_select):
                        rslt_performance.append(dict(kwargs, **value))
        df_sub_per = pd.DataFrame(rslt_performance).fillna("-")
        rslt["PerformanceSubGroup"] = df_sub_per

        return rslt
