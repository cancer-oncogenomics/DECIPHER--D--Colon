#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/7 15:20

import h2o
from h2o.estimators.xgboost import H2OXGBoostEstimator as H2oGX

from estimators.estimators_base import H2oEstimators

__all__ = ["H2OXGBoost"]


class H2OXGBoost(H2oEstimators):

    def __init__(self, **kwargs):

        super().__init__()

        self.algorithm = f"H2o--XGBoost"
        self.version_h2o = h2o.__version__
        self.model = H2oGX(keep_cross_validation_predictions=True,
                           keep_cross_validation_models=True,
                           keep_cross_validation_fold_assignment=True,
                           **kwargs)
