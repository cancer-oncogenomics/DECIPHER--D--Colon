#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/17 15:32

import os

import h2o
import pandas as pd

from module import cluster
from module.load_model import load_model
from module.save_model import save_model
from module.frame import GsFrame

__all__ = ["predict"]


def predict(f_model, feature, dataset=None, nthreads=5, max_mem_size="20000M", skip_in_model=False, f_output=None,
            submit_shell=False, precise=False):

    prefix = os.path.basename(f_model).replace(".gsml", "")

    gf_pred = GsFrame(feature_list=feature, dataset_list=dataset)
    model = load_model(f_model, use_predict=True)
    if not submit_shell:
        score = model.predict(predict_frame=gf_pred)
    else:
        score = model.predict(predict_frame=gf_pred, submit="none")

    # 深度学习模型，会返回多个结果，但是第一个肯定是预测得分
    if type(score) == tuple:
        score = score[0]

    if f_output:
        if not precise:
            score.to_csv(f_output, sep="\t", index=False)
        else:
            df_feature = pd.concat([pd.read_csv(i) for i in feature], ignore_index=True, sort=False)
            score[score.SampleID.isin(df_feature.SampleID)].to_csv(f_output, sep="\t", index=False)

    if not skip_in_model:
        save_model(model=model, path=os.path.dirname(f_model), prefix=prefix, skip_h2o=True)
    print(score)
