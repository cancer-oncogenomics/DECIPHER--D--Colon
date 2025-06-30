#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/10 10:28

"""h2o automl训练与预测流程"""

import logging

import coloredlogs

from module import cluster
from module.frame import GsFrame
from estimators.automl import H2OAutoML


__all__ = ["pipe_h2o_automl"]


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def pipe_h2o_automl(train_info, feature, pred_info=None, leaderboard: list = None, d_output=None, prefix=None,
                    blending=None, weights_column=None, nthreads=10,  **kwargs):
    """"""

    # 初始化h2o server
    logger.info(f"connect h2o server. <nthreads: {nthreads}; max_mem_size: {nthreads * 4 * 1000}M>")
    cluster.init(nthreads=nthreads, max_mem_size=f"{nthreads * 4 * 1000}M")

    # 生成数据集
    logger.info(f"generate GsFrame...")
    gf_train = GsFrame(dataset_list=train_info, feature_list=feature)
    gf_pred = GsFrame(dataset_list=pred_info, feature_list=feature) if pred_info else None
    leaderboard_frame = GsFrame(dataset_list=leaderboard, feature_list=feature) if leaderboard else None
    blending_frame = GsFrame(feature_list=blending) if blending else None

    # automl 训练
    logger.info(f"H2oAutoML training...")
    model = H2OAutoML(**kwargs)
    model.train(d_output=d_output,
                prefix=prefix,
                x=gf_train.c_features,
                y="Response",
                training_frame=gf_train,
                predict_frame=gf_pred,
                leaderboard_frame=leaderboard_frame,
                blending_frame=blending_frame,
                weights_column=weights_column
                )
    cluster.close()
    logger.info(f"success!")
