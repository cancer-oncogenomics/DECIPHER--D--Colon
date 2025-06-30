#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/1 14:52

import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as H2oGLM

from estimators.estimators_base import H2oEstimators


__all__ = ["H2OGeneralizedLinear"]


class H2OGeneralizedLinear(H2oEstimators):

    def __init__(self, **kwargs):

        super().__init__()

        self.algorithm = f"H2o--GeneralizedLinear"
        self.version_h2o = h2o.__version__
        self.model = H2oGLM(keep_cross_validation_predictions=True,
                            keep_cross_validation_models=True,
                            keep_cross_validation_fold_assignment=True,
                            **kwargs)
