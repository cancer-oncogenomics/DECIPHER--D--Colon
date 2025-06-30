#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/7 15:20

import h2o
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator as H2oStacked
import pandas as pd

from estimators.estimators_base import H2oEstimators

__all__ = ["H2OStackedEnsemble"]


class H2OStackedEnsemble(H2oEstimators):

    def __init__(self, base_models, **kwargs):

        super().__init__()

        self.algorithm = f"H2o--StackedEnsemble"
        self.version_h2o = h2o.__version__
        self.model = H2oStacked(keep_levelone_frame=True,
                                metalearner_params={"keep_cross_validation_predictions": True,
                                                    "keep_cross_validation_fold_assignment": True,
                                                    "keep_cross_validation_models": True,
                                                    },
                                base_models=base_models,
                                **kwargs
                                )

    def train(self,  x=None, y=None, training_frame=None, predict_frame=None, **kwargs):
        self.model.train(x=x, y=y, training_frame=training_frame.as_h2o, **kwargs)
        _json = self.model.metalearner()._model._model_json["output"]

        df_score = h2o.get_frame(_json["cross_validation_holdout_predictions_frame_id"]["name"]).as_data_frame()
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
        return df_score
