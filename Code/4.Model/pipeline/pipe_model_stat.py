#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/17 9:43

"""统计单个模型的性能"""

from collections import defaultdict
from glob import glob
from itertools import product
import json
import logging
import os

import coloredlogs
import h2o
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from module.load_model import load_model
from model.model_base import GsModelStat
from module.dataset import GsDataset
from module import cluster

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

__all__ = ["pipe_model_stat", "PipeModelStat"]

class PipeModelStat(object):

    def __init__(self, f_model=None, f_score=None, d_output: str = None, model_name=None, dataset: dict = None,
                 spec_list: list = None, cutoff_dataset_list=None, stat_cols=None, d_base_models=None,
                 optimize: dict = None, cs_conf: dict = None, skip_auc=False, skip_performance=False,
                 skip_combine_score=False, skip_by_subgroup=False, out_var_imp=False, sens_list: list = None
                 ):
        self.f_model = f_model
        self.f_score = f_score
        self.d_output = self.outdir(d_output)
        self.model_name = model_name
        self.dataset = dataset
        self.spec_list = spec_list  or [0.85, 0.9, 0.95]
        self.sens_list = sens_list  or []
        self.cutoff_dataset_list = cutoff_dataset_list or ["Train"]
        self.stat_cols = stat_cols or ["Train_Group", "Detail_Group", "Project"]
        self.d_base_models = d_base_models
        self.skip_auc  = skip_auc
        self.skip_performance  = skip_performance
        self.skip_combine_score  = skip_combine_score
        self.skip_by_subgroup  = skip_by_subgroup
        self.out_var_imp = out_var_imp

        if f_model:
            self.model = load_model(f_model=f_model)
        else:
            self.model = GsModelStat(f_score=f_score)
        self.model.set_dataset(dataset=dataset, optimize=optimize, cs_conf=cs_conf)

    @property
    def choose_models(self):
        """被选中的基模型"""

        return {c: c.split("--")[0] for c in self.model.score.columns if "--" in c}

    @property
    def choose_models_skip_index(self):
        """被选中的基模型, 去掉模型ID后面的index"""

        return {c.rsplit(".")[0]: c.split("--")[0] for c in self.model.score.columns if "--" in c}

    @property
    def choose_features(self):
        """被选中的特征"""

        return [c for c in self.choose_models.values()]

    @property
    def all_base_models(self):
        """基模型的分类，属于哪个特征"""

        rslt = {}
        if self.d_base_models:
            for file in glob(f"{self.d_base_models}/*/*.ModelStat.AUC.tsv"):
                if not "StackedEnsemble_AllModels" in file:
                    model_id = os.path.basename(file).replace(".ModelStat.AUC.tsv", "")
                    n_feature = file.split("/")[-2]
                    rslt[f"{n_feature}-{model_id}"] = {"model_id": model_id, "feature": n_feature, "f_auc": file}
        return rslt

    def base_stat(self):
        """基本统计结果，包括AUC,performance等"""

        rslt = defaultdict(list)
        for spec, cutoff_dataset in product(self.spec_list, self.cutoff_dataset_list):
            cutoff = self.model.cutoff(spec=spec, Dataset=cutoff_dataset)
            msg = {"ModelID": self.model_name, "Spec": spec, "CutoffDataset": cutoff_dataset, "Cutoff": cutoff}
            tmp = self.model.summary(cutoff=cutoff, stat_cols=self.stat_cols, skip_auc=self.skip_auc,
                                     skip_performance=self.skip_performance,
                                     skip_combine_score=self.skip_combine_score, skip_by_subgroup=self.skip_by_subgroup,
                                     **msg)

            rslt["Auc"].append(tmp["Auc"])
            rslt["Performance"].append(tmp["Performance"])
            rslt["CombineScore"].append(tmp["CombineScore"])
            rslt["AucSubGroup"].append(tmp["AucSubGroup"])
            rslt["PerformanceSubGroup"].append(tmp["PerformanceSubGroup"])
            rslt["classify"].append(tmp["classify"])

        if self.sens_list:
            for sens, cutoff_dataset in product(self.sens_list, self.cutoff_dataset_list):
                cutoff = self.model.cutoff(sens=sens, Dataset=cutoff_dataset)
                msg = {"ModelID": self.model_name, "Spec": f"sens-{sens}", "CutoffDataset": cutoff_dataset, "Cutoff": cutoff}
                tmp = self.model.summary(cutoff=cutoff, stat_cols=self.stat_cols, skip_auc=self.skip_auc,
                                         skip_performance=self.skip_performance,
                                         skip_combine_score=self.skip_combine_score,
                                         skip_by_subgroup=self.skip_by_subgroup,
                                         **msg)

                rslt["Auc"].append(tmp["Auc"])
                rslt["Performance"].append(tmp["Performance"])
                rslt["CombineScore"].append(tmp["CombineScore"])
                rslt["AucSubGroup"].append(tmp["AucSubGroup"])
                rslt["PerformanceSubGroup"].append(tmp["PerformanceSubGroup"])
                rslt["classify"].append(tmp["classify"])


        df_auc = pd.concat(rslt["Auc"], ignore_index=True, sort=False)
        df_performance = pd.concat(rslt["Performance"], ignore_index=True, sort=False)
        df_cs = pd.concat(rslt["CombineScore"], ignore_index=True, sort=False)
        df_auc_sub = pd.concat(rslt["AucSubGroup"], ignore_index=True, sort=False)
        df_auc_performance = pd.concat(rslt["PerformanceSubGroup"], ignore_index=True, sort=False)
        df_class = pd.concat(rslt["classify"], ignore_index=True, sort=False)

        df_auc.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.AUC.tsv", index=False, sep="\t")
        df_performance.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.Performance.tsv", index=False, sep="\t")
        df_cs.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.CombineScore.tsv", index=False, sep="\t")
        df_auc_sub.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.AucSubGroup.tsv", index=False, sep="\t")
        df_auc_performance.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.PerformanceSubGroup.tsv", index=False, sep="\t")
        df_class.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.PredClassify.tsv", index=False, sep="\t")

    def base_model_auc(self):
        """基模型的auc结果"""

        rslt = []
        for data in self.all_base_models.values():
            df_t = pd.read_csv(data["f_auc"], sep="\t")
            df_t = df_t.drop_duplicates(subset="Group2")
            for _, s in df_t.iterrows():
                tmp = {"ModelID": data["model_id"], "Feature": data["feature"], "Dataset": s.Group2, "AUC": s.AUC}
                rslt.append(tmp)
        if rslt:
            df_auc = pd.DataFrame(rslt)
            df_auc["choose"] = df_auc.ModelID.apply(lambda x: "choose" if x in self.choose_models_skip_index.keys() else "-")
            df_auc = df_auc.sort_values(by=["choose", "Dataset", "Feature", "AUC"], ascending=False)
            df_auc.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.BaseModels.AUC.tsv", index=False, sep="\t")

    def var_imp(self):
        """获取模型权重"""

        f_model = self.f_model or self.f_score.replace(".Predict.tsv", ".gsml")
        if os.path.exists(f_model):
            cluster.init()
            model = load_model(f_model, use_predict=True)
            df_imp = model.varimp()
            df_imp.to_csv(f"{self.d_output}/{self.model_name}.ModelStat.VarImp.tsv", sep='\t', index=False)
            cluster.close()

    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p

    def __call__(self, *args, **kwargs):
        self.base_stat()
        if self.out_var_imp:
            try:
                self.var_imp()
            except Exception as error:
                logger.error(error)
                pass

        if self.d_base_models:
            logger.info("scale base model auc")
            try:
                self.base_model_auc()
            except Exception as error:
                logger.error(error)
                pass
            else:
                logger.info("scale base model auc done")
