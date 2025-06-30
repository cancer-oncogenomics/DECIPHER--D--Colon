#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/8/5 14:12
# @File     : ds_compare.py
# @Project  : gsml

"""比较两个数据集之间的差异"""

import os

import pandas as pd
import pymongo


class DsCompare(object):

    def __init__(self, ds1, ds2, d_output=None, prefix="ds_compare"):
        self.ds1 = ds1
        self.ds2 = ds2
        self.d_output = self.outdir(d_output) if d_output else None
        self.prefix = prefix

    def sample_dist(self):
        """两个数据集的样本分布差异"""

        # 读取文件
        df_1 = pd.read_csv(self.ds1, sep="\t", header=None, usecols=[0], names=["SampleID"])
        df_1["Dataset"] = "Dataset1"
        df_2 = pd.read_csv(self.ds2, sep="\t", header=None, usecols=[0], names=["SampleID"])
        df_2["Dataset"] = "Dataset2"

        # 确定差异样本
        inter_ids = set(df_1.SampleID) & set(df_2.SampleID)
        df_all = pd.concat([df_1, df_2], ignore_index=True, sort=False).fillna("")
        df_all = df_all[~df_all.SampleID.isin(inter_ids)]
        df_all = df_all.drop_duplicates(subset="SampleID")

        # 从数据集中拉取临床信息
        client = pymongo.MongoClient("mongodb://Mercury:cf5ed606-2e64-11eb-bbe1-00d861479082@10.1.2.171:27001/")
        conn = client["Mercury"]["mercury_final"]
        rslt = []
        for _, s in df_all.iterrows():
            data = conn.find_one({"SampleID": s.SampleID}, {"_id": 0})
            data = dict(dict(s), **data)
            rslt.append(data)
        df_diff = pd.DataFrame(rslt)

        group_cols = ["Dataset", "ProjectID", "GroupLevel1", "GroupLevel2"]
        df_compare = df_diff.groupby(group_cols).agg({
            "SampleID": "size",
            "GroupLevel3": lambda x: ";".join(set(x)),
            "GroupLevel4": lambda x: ";".join(set(x)),
        }).reset_index(drop=False).rename(columns={"SampleID": "Count"})

        if self.d_output:
            d_output = self.outdir(f"{self.d_output}/sample_dist")
            df_compare.to_csv(f"{d_output}/{self.prefix}.sample_dist.tsv", index=False, sep="\t")
            df_diff[["Dataset", "ProjectID", "SampleID"]].to_csv(f"{d_output}/{self.prefix}.diff_sample.list", sep="\t", index=False)


    def compare(self):
        """各方面全面比较"""

        # 比较样本的差异
        self.sample_dist()


    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p


if __name__ == '__main__':
    DsCompare(
        ds1="/dssg02/InternalResearch02/Mercury/MercuryDataSet/PanCancer/PanCancer_2022-08-04/info/PanCancer_2022-08-04.all.id.list",
        ds2="/dssg02/InternalResearch02/Mercury/Monthly_Presentation_Analysis/2022_06_PanCancer/InfoList5715_220713/all.ids",
        d_output="/dssg/home/sheny/test/dataset_compare",

    ).compare()
