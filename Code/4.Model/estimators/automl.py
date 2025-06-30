#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/2 18:21

import json

import h2o
from h2o.model.model_base import ModelBase
from h2o.automl import H2OAutoML as H2oML
import pandas as pd

from estimators.estimators_base import H2oEstimators
from module.save_model import save_model
from module.frame import GsFrame


__all__ = ["H2OAutoML"]


class H2OAutoML(object):

    def __init__(self, **kwargs):
        self.version_h2o = h2o.__version__
        self.model = H2oML(keep_cross_validation_predictions=True,
                           keep_cross_validation_models=True,
                           keep_cross_validation_fold_assignment=True,
                           **kwargs)

    def train(self, d_output, prefix, x=None, y=None, training_frame=None, predict_frame=None, weights_column=None,
              leaderboard_frame=None, blending_frame=None):

        leaderboard_frame = leaderboard_frame.as_h2o if leaderboard_frame else None
        self.model.train(x=x, y=y, training_frame=training_frame.as_h2o, weights_column=weights_column,
                         leaderboard_frame=leaderboard_frame, blending_frame=blending_frame)

        # 保存模型性能
        f_leaderboard = f"{d_output}/{prefix}.leaderboard.tsv"
        df_leaderboard = self.model.leaderboard.as_data_frame()
        df_leaderboard.to_csv(f_leaderboard, sep="\t", index=False)

        # 保存base model
        for model_id in df_leaderboard.model_id:
            print(f"save model <{model_id}>")
            h2o_model = h2o.get_model(model_id)

            if "StackedEnsemble" in model_id:
                model = BaseModelStacked(model=h2o_model, training_frame=training_frame)
            else:
                model = BaseModel(model=h2o_model, training_frame=training_frame)

            if predict_frame:
                model.predict(predict_frame=predict_frame)

            save_model(model, path=d_output, prefix=f"{prefix}--{h2o_model.model_id}")


class BaseModel(H2oEstimators):

    def __init__(self, model, training_frame: GsFrame):

        super().__init__()

        self.algorithm = f"H2o--AutoMLBaseModel"
        self.version_h2o = h2o.__version__
        self.model = model
        self._score = self.get_score(training_frame)
        self.training_frame = training_frame.as_pd
        self.training_features = training_frame.c_features

    def get_score(self, training_frame: GsFrame):

        df_score = self.model.cross_validation_holdout_predictions().as_data_frame()
        if "Cancer" in df_score.columns:
            df_score["Score"] = df_score.apply(lambda x: x.Cancer, axis=1)
        else:
            df_score["Score"] = -1
        df_score.insert(0, "SampleID", training_frame.samples)
        df_score["PredType"] = "train"
        return df_score


class BaseModelStacked(H2oEstimators):

    def __init__(self, model, training_frame: GsFrame):

        super().__init__()

        self.algorithm = f"H2o--AutoMLBaseModelStacked"
        self.version_h2o = h2o.__version__
        self.model = model
        self._score = self.get_score(training_frame)
        self.training_frame = training_frame.as_pd
        self.training_features = training_frame.c_features

    def get_score(self, training_frame: GsFrame):

        cv_name = self.model.metalearner()._model._model_json["output"]["cross_validation_holdout_predictions_frame_id"]["name"]
        df_score = h2o.get_frame(cv_name).as_data_frame()

        if "Cancer" in df_score.columns:
            df_score["Score"] = df_score.apply(lambda x: x.Cancer, axis=1)
        else:
            df_score["Score"] = -1
        df_score.insert(0, "SampleID", training_frame.samples)
        df_score["PredType"] = "train"
        return df_score


