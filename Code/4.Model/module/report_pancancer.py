#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/12/21 10:20
# @File     : report_pancancer.py
# @Project  : gsml

import json
import os
import re
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *
import pymongo
from pandas.api.types import CategoricalDtype
import seaborn as sns

from model.model_base import GsModelStat


class ReportPanCancerModel(object):

    def __init__(self, f_score, f_dataset, d_output, prefix):
        self.f_score = f_score
        self.ds = pd.read_csv(f_dataset, sep="\t")
        self.d_output = self._outdir(d_output)
        self.prefix = prefix

        self.client = pymongo.MongoClient("mongodb://Mercury:cf5ed606-2e64-11eb-bbe1-00d861479082@10.1.2.171:27001/")
        self.color_list = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"] * 12

        self.conf = []

    def stat_project(self):
        """统计各个数据集的项目分布"""

        # 获取各个项目的分组（optimize or regular）
        pj_group = {}
        conn = self.client["Mercury"]["mercury_webcache"]
        for pj in conn.find_one({"name": "project_classify_regular"})["data"].split("||"):
            pj_group[pj] = "regular"
        for pj in conn.find_one({"name": "project_classify_optimize"})["data"].split("||"):
            pj_group[pj] = "optimize"

        # 获取各个项目的来源
        pj_source = {}
        conn = self.client["Mercury"]["mercury_project"]
        for pj in conn.find():
            pj_source[pj["ProjectID"]] = pj["Description"]

        rslt = []
        for ds, df_t in self.ds.groupby("Dataset"):

            df_tt = df_t.groupby(["GroupLevel1", "GroupLevel2", "ProjectID"]).size().reset_index()
            df_tt = df_tt.rename(columns={0: "Count"})
            df_tt["Group"] = df_tt.ProjectID.apply(lambda x: pj_group.get(x, "-"))
            df_tt["Source"] = df_tt.ProjectID.apply(lambda x: pj_source.get(x, "-"))
            df_tt = df_tt.sort_values(by=["Group", "GroupLevel1", "GroupLevel2", "Count"], ascending=[False, True, True, False])

            f_output = f"{self.d_output}/{self.prefix}.stat_project.{ds}.tsv"
            df_tt.to_csv(f_output, sep="\t", index=False)
            tmp = {
                "title": f"{ds}数据集",
                "id": f"project_{ds}",
                "type": "table",
                "table": {"path": f_output, "height": 690, "limit": 20}
            }
            rslt.append(tmp)
        return rslt

    def stat_cancer(self):
        """统计癌种组成"""

        rslt = []
        for ds, df_t in self.ds.groupby("Dataset"):
            # 柱形图
            f_bar = f"{self.d_output}/{self.prefix}.stat_cancer.{ds}.bar.png"
            df_plot = df_t.groupby("GroupLevel2").size().reset_index().rename(columns={0: "Count"})
            df_plot = df_plot.sort_values(by="Count", ascending=False)
            total = df_plot.Count.sum()
            df_plot["Ratio"] = df_plot.Count.apply(lambda x: f"{round(x / total * 100, 2)}%")

            df_plot.GroupLevel2 = df_plot.GroupLevel2.astype(CategoricalDtype(list(df_plot.GroupLevel2), ordered=True))
            bar_plot = (
                    ggplot(df_plot, aes("GroupLevel2", "Count")) +
                    geom_bar(stat="identity", width=0.8, fill=self.color_list[1]) +
                    geom_text(aes(x='GroupLevel2', y='Count + 40', label='Ratio'), size=7, position=position_dodge(0.8)) +
                    theme_matplotlib() +
                    labs(x=f"{ds}(N={df_plot.Count.sum()})") +
                    theme(
                        text=element_text(family="Times New Roman"),
                        axis_text_x=element_text(angle=45, ha="right")
                    )
            )
            ggsave(bar_plot, filename=f_bar, dpi=200, width=6, height=4, units="in")

            # 饼图
            f_pie = f"{self.d_output}/{self.prefix}.stat_cancer.{ds}.pie.png"
            plt.figure(figsize=[6, 4])
            self.ds[self.ds.Dataset == ds]['GroupLevel2'].value_counts().plot.pie(autopct="%1.2f")
            plt.savefig(fname=f_pie, dpi=200)

            tmp = {
                "title": f"{ds}数据集",
                "id": f"cancer_{ds}",
                "type": "plot-plot",
                "plot": {"path": f_bar},
                "plot2": {"path": f_pie},
            }
            rslt.append(tmp)
        return rslt

    def stat_tnm_blockquote(self):
        """统计分期组成的总结语"""

        df_ss = self.ds.copy()
        df_ss = df_ss[df_ss.GroupLevel1 == "Cancer"]
        df_ss.StageTnm = df_ss.StageTnm.fillna("Unknown").apply(lambda x: x.split("A")[0].split("B")[0].split("C")[0])

        rslt = round(df_ss[df_ss.StageTnm.isin(["0", "I", "II"])].shape[0] / df_ss.shape[0] * 100, 2)
        return f"早期（0,I,II期）占比~{rslt}%"

    def stat_tnm(self):
        """统计分期组成"""

        df_ss = self.ds.copy()
        df_ss = df_ss[df_ss.GroupLevel1 == "Cancer"]
        raw_tnm = list(set(df_ss.StageTnm.fillna("-")))
        df_ss.StageTnm = df_ss.StageTnm.fillna("Unknown").apply(lambda x: x.split("A")[0].split("B")[0].split("C")[0])
        df_ss.StageTnm = df_ss.StageTnm.astype(CategoricalDtype(["0", "I", "II", "III", "IV", "Unknown"]))

        rslt = []
        for ds, df_t in df_ss.groupby("Dataset"):
            # 柱形图
            f_bar = f"{self.d_output}/{self.prefix}.stat_tnm.{ds}.bar.png"
            df_plot = df_t.groupby("StageTnm").size().reset_index().rename(columns={0: "Count"})
            df_plot = df_plot.sort_values(by="Count", ascending=False)
            total = df_plot.Count.sum()
            df_plot["Ratio"] = df_plot.Count.apply(lambda x: f"{round(x / total * 100, 2)}%")
            df_plot.StageTnm = df_plot.StageTnm.astype(CategoricalDtype(list(df_plot.StageTnm), ordered=True))

            bar_plot = (
                    ggplot(df_plot, aes("StageTnm", "Count")) +
                    geom_bar(stat="identity", width=0.8, fill=self.color_list[1]) +
                    geom_text(aes(x='StageTnm', y='Count + 40', label='Ratio'), size=7, position=position_dodge(0.8)) +
                    theme_matplotlib() +
                    labs(x="") +
                    theme(
                        text=element_text(family="Times New Roman"),
                        axis_text_x=element_text(angle=45, ha="right")
                    )
            )
            ggsave(bar_plot, filename=f_bar, dpi=200, width=6, height=4, units="in")

            # 饼图
            f_pie = f"{self.d_output}/{self.prefix}.stat_tnm.{ds}.pie.png"
            plt.figure(figsize=[6, 4])
            df_plot = df_t['StageTnm'].value_counts().reset_index()
            df_plot = df_plot.sort_values(by="index")
            df_plot = df_plot.set_index("index")

            df_plot.StageTnm.plot.pie(autopct="%1.2f")
            plt.savefig(fname=f_pie, dpi=200)

            tmp = {
                "title": f"{ds}数据集",
                "id": f"cancer_{ds}",
                "type": "plot-plot",
                "plot": {"path": f_bar},
                "plot2": {"path": f_pie},
                "blockquote": f"去除健康样本后。原始分期类型有：{'|'.join(raw_tnm)}。归纳为如下6类。"
            }
            rslt.append(tmp)
        return rslt

    def stat_model_performance(self):
        """模型整体性能统计"""

        f_table = f"{self.d_output}/{self.prefix}.stat_model_performance.tsv"
        f_fig = f"{self.d_output}/{self.prefix}.stat_model_performance.png"
        color_list = ["#338FCE", "#F2CD33"]

        model = GsModelStat(f_score=self.f_score)
        model.dataset = {i: "" for i in set(self.ds.Dataset)}
        model._df_ss = self.ds.copy()

        rslt_plot = []
        rslt_table = []
        for spec in [0.85, 0.9, 0.95, 0.98, 0.99]:
            cutoff = model.cutoff(spec=spec, Dataset="Train")
            for ds in set(self.ds.Dataset):
                tmp = model.performance(cutoff=cutoff, Dataset=ds)
                rslt_table.append(dict({"Ds": ds, "TSpec": spec, "Cutoff": cutoff}, **tmp))
                rslt_plot.append({"Dataset": ds, "Spec": f"{spec}Spec", "Group": "sensitivity", "Value": tmp["sensitivity"]})
                rslt_plot.append({"Dataset": ds, "Spec": f"{spec}Spec", "Group": "specificity", "Value": tmp["specificity"]})
                rslt_plot.append({"Dataset": ds, "Spec": f"{spec}Spec", "Group": "accuracy", "Value": tmp["accuracy"]})
                rslt_plot.append({"Dataset": ds, "Spec": f"{spec}Spec", "Group": "AUC", "Value": model.auc(Dataset=ds)})

        df_table = pd.DataFrame(rslt_table)
        df_table = df_table.rename(columns={"sensitivity": "Sens", "specificity": "Spec", "accuracy": "Acc"})
        df_table.round(3).to_csv(f_table, sep="\t", index=False)

        df_plot = pd.DataFrame(rslt_plot)
        df_plot = df_plot[df_plot.Spec.isin(["0.95Spec", "0.98Spec"])]

        df_plot.Spec = df_plot.Spec.map({"0.95Spec": "0.95 Train Specificity", "0.98Spec": "0.98 Train Specificity"})
        df_plot["Group"] = df_plot.Group.astype(CategoricalDtype(["sensitivity", "specificity", "accuracy", "AUC"], ordered=True))

        base_plot = (
                ggplot(df_plot.round(3), aes("Group", "Value", fill="Dataset")) +
                geom_bar(stat='identity', position=position_dodge(0.9), width=0.7, size=0.5) +
                geom_text(aes(x='Group', y='Value + 0.025', label='Value'), size=6, position=position_dodge(0.9)) +
                scale_fill_manual(values=color_list) +
                theme_matplotlib() +
                facet_wrap("~ Spec") +
                labs(y="", x="", fill="DataSet") +
                theme(
                    text=element_text(family="Times New Roman"),
                    axis_text_x=element_text(angle=90, ha="right")
                )
        )
        ggsave(base_plot, filename=f_fig, dpi=200, width=6, height=4, units="in")

        tmp = {
            "plot": f_fig,
            "table": f_table,
        }
        return tmp

    def stat_model_performance_by_cancer(self):
        """各癌种性能"""

        report = []
        f_table = f"{self.d_output}/{self.prefix}.stat_model_performance_by_cancer.tsv"

        model = GsModelStat(f_score=self.f_score)
        model.dataset = {i: "" for i in set(self.ds.Dataset)}
        model._df_ss = self.ds.copy()

        # 各癌种性能表
        spec_list = [0.85, 0.9, 0.95, 0.98, 0.99]
        ds_list = set(self.ds.Dataset)
        cancer_list = set(self.ds.GroupLevel2)

        rslt = []
        blockquote = {}
        for spec, ds, cancer in product(*[spec_list, ds_list, cancer_list]):
            cutoff = model.cutoff(spec=spec, Dataset="Train")
            tmp = model.performance(cutoff=cutoff, Dataset=ds, GroupLevel2=cancer)
            rslt.append(dict({"Dataset": ds, "TSpec": spec, "Cancer": cancer}, **tmp))
            blockquote[str(spec)] = f"train {spec} spec cutoff = {round(cutoff, 3)}"
        df_table = pd.DataFrame(rslt)
        df_table.round(4).to_csv(f_table, sep="\t", index=False)
        report.append({
            "title": f"性能表",
            "id": f"stat_model_performance_by_cancer_table",
            "type": "table",
            "blockquote": " | ".join(blockquote.values()),
            "table": {"path": f_table, "height": 450, "limit": 10},
        })

        for ds in ds_list:
            # 各癌种性能柱形图
            f_bar = f"{self.d_output}/{self.prefix}.stat_model_performance_by_cancer.{ds}.bar.png"
            df_plot = df_table[df_table.Dataset == ds]
            df_plot = df_plot[df_plot.TSpec.isin([0.95, 0.98])]

            # 数量小于10的癌种不与展示
            cancer_list = [c for c in set(df_plot.Cancer) if self.ds[(self.ds.GroupLevel2 == c) & (self.ds.Dataset == ds)].shape[0] >= 10]
            ignore_list = [c for c in set(df_plot.Cancer) if self.ds[(self.ds.GroupLevel2 == c) & (self.ds.Dataset == ds)].shape[0] < 10]
            df_plot = df_plot[df_plot.Cancer.isin(cancer_list)]

            df_plot.TSpec = df_plot.TSpec.map({0.95: f"0.95Spec", 0.98: "0.98Spec"})
            df_plot = df_plot.sort_values(by=["specificity", "accuracy"], ascending=False)
            sort_list = list(df_plot[(df_plot.TSpec == "0.95Spec")]["Cancer"])
            df_plot.Cancer = df_plot.Cancer.astype(CategoricalDtype(sort_list, ordered=True))

            bar_plot = (
                    ggplot(df_plot.round(2), aes("Cancer", "accuracy", fill="TSpec")) +
                    geom_bar(stat='identity', position=position_dodge(0.8), width=0.7, size=0.5) +
                    geom_text(aes(x='Cancer', y='accuracy + 0.025', label='accuracy'), size=6, position=position_dodge(0.8)) +
                    theme_matplotlib() +
                    scale_fill_manual(values=["#338FCE", "#F2CD33"]) +
                    labs(y="accuracy", x="", fill="") +
                    scale_y_continuous(limits=(0, 1.1), breaks=np.arange(0, 1.1, 0.2)) +
                    theme(
                        axis_text_x=element_text(angle=45, ha="right"),
                        legend_position=(0.83, 0.85),
                        legend_key_size=8,
                        legend_text=element_text(size=8),
                    )
            )
            ggsave(bar_plot, filename=f_bar, dpi=200, width=6, height=4, units="in")

            # 得分箱线图(与柱形图的样本保持一致)
            f_box = f"{self.d_output}/{self.prefix}.stat_model_performance_by_cancer.{ds}.boxplot.png"
            df_score = pd.read_csv(self.f_score, sep="\t")
            df_score = pd.merge(df_score, self.ds, on="SampleID", how="inner")
            df_score = df_score[df_score.GroupLevel2.isin(cancer_list)]
            df_score = df_score[df_score.Dataset == ds]

            df_score.GroupLevel2 = df_score.GroupLevel2.astype(CategoricalDtype(sort_list, ordered=True))
            box_plot = (
                ggplot(df_score, aes("GroupLevel2", "Score", color="GroupLevel2")) +
                geom_boxplot(show_legend=False, outlier_size=0.001) +
                geom_jitter(width=0.2, size=1, stroke=0.1, shape='o', show_legend=False) +
                theme_matplotlib() +
                labs(x="", title="") +
                theme(
                    text=element_text(family="Times New Roman"),
                    axis_text_x=element_text(angle=45, ha="right"),

                )
            )
            ggsave(box_plot, filename=f_box, dpi=200, width=6, height=4, units="in")

            report.append({
                "title": f"性能与得分-{ds}",
                "id": f"stat_model_performance_by_cancer_plot_{ds}",
                "blockquote": f"样本数小于10的癌种不与展示：{','.join(ignore_list)}",
                "type": "plot-plot",
                "plot": {"path": f_bar},
                "plot2": {"path": f_box},
            })

        return report

    def stat_model_performance_by_tnm(self):
        """各分期性能"""

        report = []

        # 各分期性能表
        f_table = f"{self.d_output}/{self.prefix}.stat_model_performance_by_tnm.tsv"

        df_ss = self.ds.copy()
        df_ss.StageTnm = df_ss.StageTnm.fillna("Unknown")
        df_ss.StageTnm = df_ss.apply(lambda x: "Healthy" if x.GroupLevel1 == "Healthy" else re.split("A|B|C", x.StageTnm)[0], axis=1)

        model = GsModelStat(f_score=self.f_score)
        model.dataset = {i: "" for i in set(df_ss.Dataset)}
        model._df_ss = df_ss.copy()

        spec_list = [0.85, 0.9, 0.95, 0.98, 0.99]
        ds_list = set(df_ss.Dataset)
        tnm_list = [i for i in set(df_ss.StageTnm) if i != "Healthy"]

        rslt = []
        for spec, ds, tnm in product(*[spec_list, ds_list, tnm_list]):
            cutoff = model.cutoff(spec=spec, Dataset="Train")
            tmp = model.performance(cutoff=cutoff, Dataset=ds, StageTnm=tnm)
            rslt.append(dict({"Dataset": ds, "TSpec": spec, "StageTnm": tnm}, **tmp))
        df_table = pd.DataFrame(rslt)
        df_table.round(4).to_csv(f_table, sep="\t", index=False)
        report.append({
            "title": f"性能表",
            "id": f"stat_model_performance_by_tnm_table",
            "type": "table",
            "table": {"path": f_table, "height": 450, "limit": 10},
        })

        for ds in ds_list:
            # 各分期性能柱形图
            f_bar = f"{self.d_output}/{self.prefix}.stat_model_performance_by_tnm.{ds}.bar.png"
            df_plot = df_table[df_table.Dataset == ds]
            df_plot = df_plot[df_plot.TSpec.isin([0.95, 0.98])]

            # 数量小于10的癌种不与展示
            mnt_list = [c for c in set(df_plot.StageTnm) if df_ss[(df_ss.StageTnm == c) & (df_ss.Dataset == ds)].shape[0] >= 10]
            ignore_list = [c for c in set(df_plot.StageTnm) if df_ss[(df_ss.StageTnm == c) & (df_ss.Dataset == ds)].shape[0] < 10]
            df_plot = df_plot[df_plot.StageTnm.isin(mnt_list)]

            df_plot.TSpec = df_plot.TSpec.map({0.95: f"0.95Spec", 0.98: "0.98Spec"})
            df_plot = df_plot.sort_values(by=["StageTnm"])
            sort_list = list(df_plot[(df_plot.TSpec == "0.95Spec")]["StageTnm"])
            df_plot.StageTnm = df_plot.StageTnm.astype(CategoricalDtype(sort_list, ordered=True))

            bar_plot = (
                    ggplot(df_plot.round(2), aes("StageTnm", "accuracy", fill="TSpec")) +
                    geom_bar(stat='identity', position=position_dodge(0.8), width=0.7, size=0.5) +
                    geom_text(aes(x='StageTnm', y='accuracy + 0.025', label='accuracy'), size=6, position=position_dodge(0.8)) +
                    theme_matplotlib() +
                    scale_fill_manual(values=["#338FCE", "#F2CD33"]) +
                    labs(y="accuracy", x="", fill="") +
                    scale_y_continuous(limits=(0, 1.1), breaks=np.arange(0, 1.1, 0.2)) +
                    theme(
                        axis_text_x=element_text(angle=45, ha="right"),
                        legend_position=(0.83, 0.85),
                        legend_key_size=8,
                        legend_text=element_text(size=8),
                    )
            )
            ggsave(bar_plot, filename=f_bar, dpi=200, width=6, height=4, units="in")

            # 得分箱线图(与柱形图的样本保持一致)
            f_box = f"{self.d_output}/{self.prefix}.stat_model_performance_by_tnm.{ds}.boxplot.png"
            df_score = pd.read_csv(self.f_score, sep="\t")
            df_score = pd.merge(df_score, df_ss, on="SampleID", how="inner")
            df_score = df_score[df_score.StageTnm.isin(tnm_list)]
            df_score = df_score[df_score.Dataset == ds]

            df_score.StageTnm = df_score.StageTnm.astype(CategoricalDtype(sort_list, ordered=True))
            box_plot = (
                    ggplot(df_score, aes("StageTnm", "Score", color="StageTnm")) +
                    geom_boxplot(show_legend=False, outlier_size=0.001) +
                    geom_jitter(width=0.2, size=1, stroke=0.1, shape='o', show_legend=False) +
                    theme_matplotlib() +
                    labs(x="", title="") +
                    theme(
                        text=element_text(family="Times New Roman"),
                        axis_text_x=element_text(angle=45, ha="right"),
                    )
            )
            ggsave(box_plot, filename=f_box, dpi=200, width=6, height=4, units="in")

            report.append({
                "title": f"性能与得分-{ds}",
                "id": f"stat_model_performance_by_cancer_plot_{ds}",
                "blockquote": f"样本数小于10的分期不与展示：{','.join(ignore_list)}",
                "type": "plot-plot",
                "plot": {"path": f_bar},
                "plot2": {"path": f_box},
            })

        return report

    def stat_model_performance_by_project(self):
        """各个项目性能"""

        report = []

        # 各项目性能表
        f_table = f"{self.d_output}/{self.prefix}.stat_model_performance_by_project.tsv"

        model = GsModelStat(f_score=self.f_score)
        model.dataset = {i: "" for i in set(self.ds.Dataset)}
        model._df_ss = self.ds.copy()

        spec_list = [0.9, 0.95, 0.98]
        rslt = []

        for spec in spec_list:
            cutoff = model.cutoff(spec=spec, Dataset="Train")
            for (ds, pj, cancer), _ in self.ds.groupby(["Dataset", "ProjectID", "GroupLevel2"]):
                print(ds, pj, cancer)
                tmp = model.performance(cutoff=cutoff, Dataset=ds, ProjectID=pj, GroupLevel2=cancer)
                rslt.append(dict({"Dataset": ds, "TSpec": spec, "CancerType": cancer, "ProjectID": pj}, **tmp))
        df_table = pd.DataFrame(rslt)
        df_table.round(4).to_csv(f_table, sep="\t", index=False)
        report.append({
            "title": f"性能表",
            "id": f"stat_model_performance_by_project_table",
            "type": "table",
            "table": {"path": f_table, "height": 450, "limit": 10},
        })

        return report

    def stat_distribute_by_age(self):
        """各数据集，年龄分布"""

        df_ss = self.ds.copy()
        df_total = df_ss.groupby("GroupLevel2").size().reset_index().rename(columns={0: "Total"})
        df_ss = pd.merge(df_ss, df_total, on="GroupLevel2", how="inner")
        df_ss["Cancer"] = df_ss.apply(lambda x: f"{x.GroupLevel2}(N={x.Total})", axis=1)

        # 删除未知样本
        df_ss = df_ss[df_ss.Age != 0]

        # 按样本总数排序
        df_ss = df_ss.sort_values(by="Total", ascending=False)
        df_ss.Cancer = df_ss.Cancer.astype(CategoricalDtype(df_ss.Cancer.drop_duplicates(), ordered=True))

        # 画图
        fig = plt.figure(figsize=(12, 5), dpi=100)
        p = sns.violinplot(x="Cancer", y="Age", hue="Dataset", data=df_ss, inner="quartile", split=True, linewidth=1,
                           colors=["#E64B35", "#4DBBD5"])
        plt.xlabel("")
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.gcf().subplots_adjust(bottom=0.3)

        f_fig = f"{self.d_output}/{self.prefix}.stat_distribute_by_age.violinplot.png"
        plt.savefig(f_fig)
        return f_fig

    def stat_distribute_by_sex(self):
        """年龄分布"""

        df_ss = self.ds.copy()

        # 去除前列腺女性样本
        df_ss = df_ss[~ ((df_ss.GroupLevel2 == "Prostate") & (df_ss.AnalyzedSex == "F"))]

        df_total = df_ss.groupby(["GroupLevel2"]).size().reset_index().rename(columns={0: "Total"})
        df_ss = df_ss.groupby(["Dataset", "GroupLevel2", "AnalyzedSex"]).size().reset_index().rename(columns={0: "Count"})
        df_plot = pd.merge(df_ss, df_total, on="GroupLevel2", how="inner")
        df_plot["Cancer"] = df_plot.apply(lambda x: f"{x.GroupLevel2}(N={x.Total})", axis=1)

        # 按样本总数排序
        df_plot = df_plot.sort_values(by="Total", ascending=False)
        df_plot.Cancer = df_plot.Cancer.astype(CategoricalDtype(df_plot.Cancer.drop_duplicates(), ordered=True))

        base_plot = (ggplot(df_plot, aes('Dataset', 'Count', fill='AnalyzedSex')) +
                     geom_bar(stat='identity', color='black', position='fill', size=0.25) +
                     theme_matplotlib() +
                     facet_wrap('~ Cancer', ncol=5) +
                     labs(x="", y="Ratio of Sex", fill="Sex") +
                     scale_fill_manual(values=self.color_list) +
                     theme(
                         text=element_text(family="Times New Roman", size=10),

                     )
                     )
        f_fig = f"{self.d_output}/{self.prefix}.stat_distribute_by_sex.boxplot.png"
        ggsave(base_plot, filename=f_fig, dpi=200, width=10, height=8, units="in")
        return f_fig

    def stat_distribute_by_tube(self):
        """采血管分布情况"""

        df_ss = self.ds.copy()
        df_ss = df_ss.fillna("Unknown")

        df_total = df_ss.groupby(["GroupLevel2"]).size().reset_index().rename(columns={0: "Total"})
        df_ss = df_ss.groupby(["Dataset", "GroupLevel2", "TubeType"]).size().reset_index().rename(columns={0: "Count"})
        df_plot = pd.merge(df_ss, df_total, on="GroupLevel2", how="inner")
        df_plot["Cancer"] = df_plot.apply(lambda x: f"{x.GroupLevel2}(N={x.Total})", axis=1)

        # 按样本总数排序
        df_plot = df_plot.sort_values(by="Total", ascending=False)
        df_plot.Cancer = df_plot.Cancer.astype(CategoricalDtype(df_plot.Cancer.drop_duplicates(), ordered=True))

        base_plot = (ggplot(df_plot, aes('Dataset', 'Count', fill='TubeType')) +
                     geom_bar(stat='identity', color='black', position='fill', size=0.25) +
                     theme_matplotlib() +
                     facet_wrap('~ Cancer', ncol=5) +
                     labs(x="", y="Ratio of TubeType") +
                     scale_fill_manual(values=self.color_list) +
                     theme(
                         text=element_text(family="Times New Roman", size=10),

                     )
                     )
        f_fig = f"{self.d_output}/{self.prefix}.stat_distribute_by_tube.bar.png"
        ggsave(base_plot, filename=f_fig, dpi=200, width=10, height=8, units="in")
        return f_fig

    @staticmethod
    def _outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p

    def __call__(self, *args, **kwargs):

        # 数据集展示
        # # 1. 项目组成
        # data = {
        #     "title": "项目组成",
        #     "id": "project",
        #     "spread": False,
        #     "blockquote": f"测试测试",
        #     "children": self.stat_project()
        # }
        # self.conf.append(data)
        #
        # # 2. 癌种组成
        # data = {
        #     "title": "癌种组成",
        #     "id": "stat_cancer",
        #     "spread": True,
        #     "children": self.stat_cancer()
        # }
        # self.conf.append(data)

        # # 分期组成
        # data = {
        #     "title": "分期组成",
        #     "id": "stat_tnm",
        #     "spread": True,
        #     "blockquote": self.stat_tnm_blockquote(),
        #     "children": self.stat_tnm()
        # }
        # self.conf.append(data)

        # # 年龄分布情况
        # data = {
        #     "title": "年龄分布",
        #     "id": "stat_distribute_by_age",
        #     "spread": True,
        #     "blockquote": "年龄分布",
        #     "type": "plot",
        #     "plot": {"path": self.stat_distribute_by_age()},
        # }
        # self.conf.append(data)
        #
        # # 性别分布情况
        # data = {
        #     "title": "性别分布",
        #     "id": "stat_distribute_by_sex",
        #     "spread": True,
        #     "blockquote": "女性去除前列腺样本。去除性别NA的样本",
        #     "type": "plot-plot",
        #     "plot": {"path": self.stat_distribute_by_sex()},
        # }
        # self.conf.append(data)
        #
        # # 采血管分布情况
        # data = {
        #     "title": "采血管分布分布",
        #     "id": "stat_distribute_by_tube",
        #     "spread": True,
        #     "blockquote": "",
        #     "type": "plot-plot",
        #     "plot": {"path": self.stat_distribute_by_tube()},
        # }
        # self.conf.append(data)
        #
        #
        # 模型整体性能
        tmp = self.stat_model_performance()
        data = {
            "title": "模型整体性能",
            "id": "model_performance",
            "spread": True,
            "type": "plot-table",
            "blockquote": f"以Train数据集划cutoff",
            "table": {"path": tmp["table"], "height": 400, "limit": 20},
            "plot": {"path": tmp["plot"]},
        }
        self.conf.append(data)

        # # 各癌种性能
        # data = {
        #     "title": "各癌种性能",
        #     "id": "stat_model_performance_by_cancer",
        #     "spread": True,
        #     "children": self.stat_model_performance_by_cancer()
        # }
        # self.conf.append(data)

        # # 各分期性能
        # data = {
        #     "title": "各分期性能",
        #     "id": "stat_model_performance_by_tnm",
        #     "spread": True,
        #     "children": self.stat_model_performance_by_tnm()
        # }
        # self.conf.append(data)

        # # 各项目性能
        # data = {
        #     "title": "各项目性能",
        #     "id": "stat_model_performance_by_project",
        #     "spread": True,
        #     "children": self.stat_model_performance_by_project()
        # }
        # self.conf.append(data)

        # 保存配置文件
        f_conf = f"{self.d_output}/{self.prefix}.report.json"
        with open(f_conf, "w") as fw:
            fw.write(json.dumps(self.conf))

        return self.conf

    def __del__(self):
        self.client.close()


if __name__ == '__main__':
    a = ReportPanCancerModel(
        f_score="/dssg02/InternalResearch02/sheny/Mercury/2022-12-21_MercuryMeet12/Analyze/Test2/CombineModel/Stacked/combine2--griffin--MC--top10--mean.Predict.tsv",
        f_dataset="/dssg02/InternalResearch02/sheny/Mercury/2022-12-21_MercuryMeet12/Analyze/Test2/Info/info/Test2.all.info.list",
        d_output="/dssg/home/sheny/test/ReportPanCancer",
        prefix="test"

    )
    b = a()
    print(b)
