#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/7 15:30


"""不同的模型实例，对应不同的train predict方法"""

import logging

import coloredlogs
import h2o
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import typing as t
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.model_base import GsModelStat
from module.error import ColumnsInconsistent
from module.preprocess import MinMaxScale
from module.frame import GsFrame
from version import __version__


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


__all__ = ["H2oEstimators", "PytorchEstimators"]


class H2oEstimators(GsModelStat):
    """ 一个Gs模型实例"""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.args = kwargs
        self.model = None
        self.version_gsml = __version__
        self._score = pd.DataFrame(columns=["SampleID", "PredType", "Score"])
        self.train_frame = None
        self.training_features = []

    def train(self, x=None, y=None, training_frame=None, predict_frame=None, **kwargs):
        """主要要得到一个score的data frame"""

        # 记录训练特征
        # self.train_frame = train_frame.as_pd
        self.training_features = training_frame.c_features

        self.model.train(x=x, y=y, training_frame=training_frame.as_h2o, **kwargs)
        df_score = self.model.cross_validation_holdout_predictions().as_data_frame()
        if "Cancer" in df_score.columns:
            df_score["Score"] = df_score.apply(lambda x: x.Cancer, axis=1)
        else:
            df_score["Score"] = -1
        df_score.insert(0, "SampleID", training_frame.samples)

        df_score["PredType"] = "train"
        self._score = df_score

        if predict_frame:
            self.predict(predict_frame=predict_frame)

    def predict(self, predict_frame):

        # 验证特征是否与训练一致
        pred_cols = predict_frame.c_features
        try:
            train_cols = self.training_features
        except:
            train_cols = predict_frame.c_features

        if set(train_cols) - set(pred_cols):
            raise ColumnsInconsistent(f"pred features columns not same as train features. {set(train_cols) - set(pred_cols)}")

        df_score = self.model.predict(predict_frame.as_h2o).as_data_frame()
        if "Cancer" in df_score.columns:
            df_score["Score"] = df_score.apply(lambda x: x.Cancer, axis=1)
        else:
            df_score["Score"] = -1
        df_score.insert(0, "SampleID", predict_frame.samples)
        df_score["PredType"] = "predict"

        train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
        df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()
        self._score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
        self._score = self._score.drop_duplicates(subset=["SampleID"], keep="last")
        return self._score

    def varimp(self, method="mean"):

        if "base_models" in dir(self.model):
            df_coef = h2o.get_model(self.model.metalearner().model_id).varimp(use_pandas=True).set_index('variable')
            imp_dict = df_coef.to_dict(orient='index')

            rslt = []
            for model_id in self.model.base_models:
                imp = imp_dict[model_id]["percentage"]
                df_t = h2o.get_model(model_id).varimp(use_pandas=True)
                df_t["relative_importance"] = df_t["relative_importance"] * imp
                df_t["scaled_importance"] = df_t["scaled_importance"] * imp
                df_t["percentage"] = df_t["percentage"] * imp
                rslt.append(df_t)

            df_imp = pd.concat(rslt, ignore_index=True, sort=False)
            df_imp = df_imp.groupby(["variable"]).sum().reset_index()
            df_imp = df_imp.sort_values(by="percentage", ascending=False)
        else:
            df_imp = self.model.varimp(use_pandas=True)

        return df_imp
