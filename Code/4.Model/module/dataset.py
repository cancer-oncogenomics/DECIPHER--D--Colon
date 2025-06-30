#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/19 13:41

import os

import pandas as pd
import yaml

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)

__all__ = ["GsDataset"]


class GsDataset(object):
    """"标准数据集信息实例"""

    def __init__(self, f_conf):

        self.conf = yaml.load(open(f_conf), Loader=yaml.FullLoader)

        self.dataset = self.conf.get("dataset", {})
        self.optimize = self.conf.get("optimize", {})
        self.feature = self.conf.get("feature", {})
        self.cs_conf = self.conf.get("combine_score")

        # self.data = self._data()

    def _data(self):
        """合并数据集和优化项目，完整的样本信息表（有重复）"""

        rslt = []
        for name, file in self.dataset.items():
            df_t = pd.read_csv(f"{self.path}/{file}", sep="\t")
            df_t.insert(0, "Dataset", name)
            rslt.append(df_t)
        df_dataset = pd.concat(rslt, ignore_index=True, sort=False)

        rslt = []
        for name, file in self.optimize.items():
            df_t = pd.read_csv(f"{self.path}/{file}", sep="\t")
            rslt.append(df_t)
        df_optimize = pd.concat(rslt, ignore_index=True, sort=False)

        df_data = pd.merge(df_dataset, df_optimize, on="SampleID", how="outer", suffixes=["", "_y"])
        return df_data

    def summary(self):
        rslt = []
        for group, df_g in self.data.groupby(["Dataset", "Train_Group", "Project"]):
            series = {"Dataset": group[0], "Train_Group": group[1], "Project": group[2],
                      "Num": len(set(df_g["SampleID"])), "Describe": self.conf["describe"].get(group[2])}
            rslt.append(series)

        df_stat = pd.DataFrame(rslt)
        return df_stat


if __name__ == '__main__':
    a = GsDataset(f_conf="/dssg/home/sheny/database/soft/gsml/2022-05-19/dataset.yaml")
    print(a.summary())
