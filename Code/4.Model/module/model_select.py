#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/8/31 10:43
# @File     : model_select.py
# @Project  : gsml

"""根据模型各项指标，筛选出符合条件的模型"""

import os
import logging

import pandas as pd
import yaml
from glob import glob

import coloredlogs
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class ModelSelect(object):

    def __init__(self, d_model, d_model_stat, f_conf, f_output, threads=10):
        self.d_model = d_model
        self.d_model_stat = d_model_stat
        self.f_conf = f_conf
        self.f_output = f_output
        self.threads = threads

    def selected(self):

        # 确定模型和统计结果
        logger.info("Determine the model and statistical results")
        rslt = []
        for f_model in glob(f"{self.d_model}/*.gsml"):
            model_id = os.path.basename(f_model).rsplit(".", 1)[0]
            tmp = {
                "ModelID": model_id,
                "f_model": f_model,
                "AUC": f"{self.d_model_stat}/{model_id}.ModelStat.AUC.tsv",
                "AucSubGroup": f"{self.d_model_stat}/{model_id}.ModelStat.AucSubGroup.tsv",
                "CombineScore": f"{self.d_model_stat}/{model_id}.ModelStat.CombineScore.tsv",
                "PerformanceSubGroup": f"{self.d_model_stat}/{model_id}.ModelStat.PerformanceSubGroup.tsv",
                "Performance": f"{self.d_model_stat}/{model_id}.ModelStat.Performance.tsv",
            }
            rslt.append(tmp)
        df_info = pd.DataFrame(rslt)

        conf = yaml.load(open(self.f_conf), Loader=yaml.FullLoader)

        # 根据筛选条件获得各个模型对应的值
        logger.info("The corresponding values of each model were obtained according to the filtering conditions")
        for name, method in conf["select_method"].items():
            rslt = Parallel(n_jobs=self.threads)(delayed(self.selected_value)(s, name, method) for _, s in df_info.iterrows())
            df_info = pd.DataFrame(rslt)

        # 确定符合条件的样本
        logger.info("Determine the models that meet the criteria")
        select_models = {}
        for name, method in conf["select_method"].items():
            df_select = df_info.copy()
            df_select = df_select.sort_values(by=name, ascending=method["ascending"])
            df_select = df_select.iloc[0: method["count"]]

            for model_id in df_select.ModelID:
                if model_id in select_models.keys():
                    select_models[model_id] = select_models[model_id] + "," + name
                else:
                    select_models[model_id] = name
        df_info["selected"] = df_info.ModelID.apply(lambda x: select_models[x] if x in select_models.keys() else "-")
        df_info = df_info.sort_values(by="selected", ascending=False)
        end_cols = ["f_model", "AUC", "AucSubGroup", "CombineScore", "PerformanceSubGroup", "Performance"]
        df_info = df_info[[c for c in df_info.columns if c not in end_cols] + end_cols]

        # 保存结果
        logger.info("Save the result")
        df_info.to_csv(f"{self.f_output}", sep="\t", index=False)

        logger.info("Done")

    @staticmethod
    def selected_value(series, name, method):
        """ 根据筛选条件，获得模型对应的结果

        :param series: df_info的每一行信息
        :param name: 筛选方法的名称
        :param method: 具体的筛选规则
        :return:
        """

        series = dict(series)
        df_stat = pd.read_csv(series[method["file"]], sep="\t")
        for k, v in method["selected"].items():
            df_stat = df_stat[df_stat[k].isin(v)]
        value = df_stat.iloc[0][method["target_value"]]
        series[name] = value
        return series



if __name__ == '__main__':
    ModelSelect(
        d_model="/dssg02/InternalResearch02/sheny/Mercury/2022-08-30_5715TrainByDbFeature/Analyze/AutoML-5715-NewFeature/cnv/",
        d_model_stat="/dssg02/InternalResearch02/sheny/Mercury/2022-08-30_5715TrainByDbFeature/Analyze/AutoML-5715-NewFeature/cnv/ModelStat/",
        f_conf="/dssg/home/sheny/MyProject/gsml/config/model_select/train_top2-valid_top2.yaml",
        f_output="/dssg/home/sheny/test/cnv.ModelSelect.tsv"
    ).selected()