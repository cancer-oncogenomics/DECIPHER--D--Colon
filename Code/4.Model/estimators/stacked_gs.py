#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/11 15:17


"""本部自己做的stacked算法"""

import os
import subprocess
import uuid
from glob import glob
from functools import reduce

import pandas as pd

from estimators.estimators_base import H2oEstimators
from estimators.glm import H2OGeneralizedLinear
from module.load_model import load_model
from module.save_model import save_model
from module.frame import GsFrame
from module.error import ColumnsInconsistent
from module.submit_lsf import submit_lsf


__all__ = ["GsStacked"]


class GsStacked(H2oEstimators):
    """ 用H2o模型生成

    :param metalearner_algorithm: stacked算法 [glm, mean]
    """

    def __init__(self, base_models, d_output, model_id="GsStacked", metalearner_algorithm="mean"):

        super().__init__()

        self.model = None
        self.metalearner_algorithm = metalearner_algorithm
        self.d_output = d_output
        self.algorithm = f"Gs--GsStacked"
        self.base_models = base_models  # 模型实例

        self.d_base_models = self.outdir(f"{d_output}/BaseModel.{model_id}")

    def create_score_frame(self, frame=None, predict=False, submit="lsf"):
        """生成关于模型得分的特征"""

        d_tmp = self.outdir(os.path.expanduser("~") + f"/.log/gsml/{uuid.uuid1()}")
        f_gsml = os.path.join(os.path.dirname(__file__), "../gsml")
        f_feature = f"{d_tmp}/feature.csv"
        frame.feature.to_csv(f_feature, index=False)

        # base model预测，得到预测文件
        pred_files = {}
        cmd_list = []

        for model_id in self.base_models:
            f_model = f"{self.d_base_models}/{model_id}.gsml"

            if predict:
                f_pred = f"{d_tmp}/{model_id}.Predict.tsv"
                cmd = f"{f_gsml} Predict -i {f_model} --feature {f_feature} -n 10 -o {f_pred} --skip_in_model"
                cmd_list.append((f"predict_{model_id}", cmd))
                pred_files[model_id] = f_pred
            else:
                f_pred = f"{self.d_base_models}/{model_id}.Predict.tsv"
                pred_files[model_id] = f_pred

        number = len([i for i in open(f_feature)])
        if cmd_list and submit == "lsf":
            nthreads = min(10, (number // 1000) + 1)  # 每1000个样本一个线程
            submit_lsf(cmd_list, d_output=d_tmp, nthreads=nthreads, wait=1)
        elif cmd_list and submit != "lsf":
            for _, cmd in cmd_list:
                subprocess.check_output(cmd, shell=True)

        # 合并预测结果，得到特征矩阵
        rslt = []
        for model_id in self.base_models:

            f_score = pred_files[model_id]
            df_score = pd.read_csv(f_score, sep="\t")
            df_score = df_score[df_score.SampleID.isin(frame.samples)]

            df_score = df_score[["SampleID", "Score"]]
            df_score = df_score.rename(columns={"Score": model_id})
            rslt.append(df_score)
        df_feature = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="inner"), rslt)

        score_frame = GsFrame()
        score_frame.dataset = frame.dataset
        score_frame.feature = df_feature
        score_frame.data = score_frame._data()

        if os.path.exists(d_tmp):
            subprocess.check_output(f"rm -r {d_tmp}", shell=True)

        return score_frame

    def train(self, x=None, y=None, training_frame=None, predict_frame=None, train_info=None, pred_info=None, re_pred=False,
              **kwargs):
        """模型训练"""

        # 将模型保存至base_model目录
        model_ids = []
        for name, model in self.base_models:
            save_model(model=model, path=self.d_base_models, prefix=name)
            model_ids.append(name)
        self.base_models = model_ids

        # 生成关于模型得分的特征
        score_frame = self.create_score_frame(training_frame, predict=False)
        self.training_frame = score_frame.as_pd
        self.training_features = score_frame.c_features

        # 进行模型训练
        df_score = score_frame.feature.copy()
        if self.metalearner_algorithm == "mean":
            df_score["Score"] = df_score[score_frame.c_features].mean(axis=1)
            df_score["PredType"] = "train"
        elif self.metalearner_algorithm == "glm":
            self.algorithm = f"H2o--GsStacked"
            model = H2OGeneralizedLinear(**kwargs)
            model.train(x=score_frame.c_features, y="Response", training_frame=score_frame)
            df_score = pd.merge(df_score, model._score, on="SampleID", how="inner")
            self.model = model.model

        self._score = df_score

        if predict_frame:
            self.predict(predict_frame, re_pred=re_pred)

    def predict(self, predict_frame, re_pred=True, submit="lsf"):
        """根据构建的stacked，预测得分"""

        # 生成关于模型得分的特征
        score_frame = self.create_score_frame(predict_frame, predict=re_pred, submit=submit)

        # 验证特征是否与训练一致
        pred_cols = score_frame.c_features
        try:
            train_cols = self.training_features
        except:
            train_cols = predict_frame.c_features
        if set(train_cols) - set(pred_cols):
            raise ColumnsInconsistent(f"pred features columns not same as train features. {set(train_cols) - set(pred_cols)}")

        # 得到预测得分：
        df_score = score_frame.feature.copy()
        if self.metalearner_algorithm == "mean":
            df_score["Score"] = df_score[score_frame.c_features].mean(axis=1)
            df_score["PredType"] = "predict"
        elif self.metalearner_algorithm == "glm":

            df_pred = self.model.predict(score_frame.as_h2o).as_data_frame()
            if "Cancer" in df_pred.columns:
                df_pred["Score"] = df_pred.apply(lambda x: x.Cancer, axis=1)
            else:
                df_pred["Score"] = -1
            df_pred.insert(0, "SampleID", score_frame.samples)
            df_pred["PredType"] = "predict"

            df_score = pd.merge(df_score, df_pred, on="SampleID", how="inner")
            df_score["PredType"] = "predict"

        train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
        df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()
        self._score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
        self._score = self._score.drop_duplicates(subset=["SampleID"], keep="last")
        return self._score

    def varimp(self, method="mean"):

        rslt = []
        for model_id in self.base_models:
            f_model = f"{self.d_base_models}/{model_id}.gsml"
            _model = load_model(f_model, use_predict=True)
            df_t = _model.varimp(method=method)
            df_t.insert(0, "ModelID", model_id)
            df_t.insert(1, "ModelGroup", "base")
            rslt.append(df_t)

        df_imp = pd.concat(rslt, ignore_index=True, sort=False)
        if method == "mean":
            df_mean = df_imp.groupby("variable")[["relative_importance", "scaled_importance", "percentage"]].agg("mean").reset_index()
            df_mean.insert(0, "ModelID", "stacked")
            df_mean.insert(1, "ModelGroup", "stacked")
        else:
            df_mean = pd.DataFrame()

        df_imp = pd.concat([df_mean, df_imp], ignore_index=True, sort=False)
        return df_imp

    @staticmethod
    def outdir(p):

        if not os.path.exists(p):
            os.makedirs(p)
        return p


class GsStackedToo(H2oEstimators):
    """ 用H2o模型生成

    :param metalearner_algorithm: stacked算法 [glm, mean]
    """

    def __init__(self, base_models, d_output, model_id="GsStacked", metalearner_algorithm="mean",
                 too_class=None):

        super().__init__()

        self.model = None
        self.metalearner_algorithm = metalearner_algorithm
        self.d_output = d_output
        self.algorithm = f"Gs--GsStacked--Too"
        self.base_models = base_models  # 模型实例
        self.too_class=too_class.split(",") if too_class else None

        self.d_base_models = self.outdir(f"{d_output}/BaseModel.{model_id}")

    def create_score_frame(self, frame=None, predict=False, submit="lsf"):
        """生成关于模型得分的特征"""

        d_tmp = self.outdir(os.path.expanduser("~") + f"/.log/gsml/{uuid.uuid1()}")
        f_gsml = os.path.join(os.path.dirname(__file__), "../gsml")
        f_feature = f"{d_tmp}/feature.csv"
        frame.feature.to_csv(f_feature, index=False)

        # base model预测，得到预测文件
        pred_files = {}
        cmd_list = []

        for model_id in self.base_models:
            f_model = f"{self.d_base_models}/{model_id}.gsml"

            if predict:
                f_pred = f"{d_tmp}/{model_id}.Predict.tsv"
                cmd = f"{f_gsml} Predict -i {f_model} --feature {f_feature} -n 10 -o {f_pred} --skip_in_model"
                cmd_list.append((f"predict_{model_id}", cmd))
                pred_files[model_id] = f_pred
            else:
                f_pred = f"{self.d_base_models}/{model_id}.Predict.tsv"
                pred_files[model_id] = f_pred

        number = len([i for i in open(f_feature)])
        if cmd_list and submit == "lsf":
            nthreads = min(10, (number // 1000) + 1)  # 每1000个样本一个线程
            submit_lsf(cmd_list, d_output=d_tmp, nthreads=nthreads, wait=1)
        elif cmd_list and submit != "lsf":
            for _, cmd in cmd_list:
                subprocess.check_output(cmd, shell=True)

        # 合并预测结果，得到特征矩阵
        rslt = []
        for model_id in self.base_models:
            f_score = pred_files[model_id]
            df_score = pd.read_csv(f_score, sep="\t")
            df_score = df_score[df_score.SampleID.isin(frame.samples)]

            df_score["ModelID"] = model_id
            rslt.append(df_score)
        df_feature = pd.concat(rslt, ignore_index=True, sort=False)

        if os.path.exists(d_tmp):
            subprocess.check_output(f"rm -r {d_tmp}", shell=True)

        return df_feature

    def train(self, x=None, y=None, training_frame=None, predict_frame=None, train_info=None, pred_info=None,
              re_pred=False, too_class=None, **kwargs):
        """模型训练"""

        # 将模型保存至base_model目录
        model_ids = []
        for name, model in self.base_models:
            save_model(model=model, path=self.d_base_models, prefix=name)
            model_ids.append(name)
        self.base_models = model_ids

        # 生成关于模型得分的特征
        df_feature = self.create_score_frame(training_frame, predict=False)
        model_count = set(df_feature.groupby(["SampleID"])[self.too_class].size())
        if len(model_count) != 1:
            raise ColumnsInconsistent(f"There are some samples of the base model did not score")

        # 进行模型训练
        if self.metalearner_algorithm == "mean":
            df_score = df_feature.groupby(["SampleID"])[self.too_class].mean().reset_index()
            df_score["PredType"] = "train"
            df_score["Top1Class"] = df_score[self.too_class].idxmax(1)
            df_score["Top2Class"] = df_score[self.too_class].apply(lambda j: j.sort_values(ascending=False).index[1], axis=1)
            df_score["Top3Class"] = df_score[self.too_class].apply(lambda j: j.sort_values(ascending=False).index[2], axis=1)
            df_score["AllClass"] = df_score[self.too_class].apply(lambda j: ",".join(j.sort_values(ascending=False).index), axis=1)
            df_score["AllScore"] = df_score[self.too_class].apply(lambda j: ",".join(j.round(4).sort_values(ascending=False).astype(str)), axis=1)
        else:
            raise ValueError("The TOO model does not support algorithms other than mean")

        self._score = df_score

        if predict_frame:
            self.predict(predict_frame, re_pred=re_pred)

    def predict(self, predict_frame, re_pred=True, submit="lsf"):
        """根据构建的stacked，预测得分"""

        # 生成关于模型得分的特征
        df_feature = self.create_score_frame(predict_frame, predict=re_pred, submit=submit)

        # 验证特征是否与训练一致
        model_count = set(df_feature.groupby(["SampleID"])[self.too_class].size())
        if len(model_count) != 1:
            raise ColumnsInconsistent(f"There are some samples of the base model did not score")

        # 得到预测得分：
        if self.metalearner_algorithm == "mean":
            df_score = df_feature.groupby(["SampleID"])[self.too_class].mean().reset_index()
            df_score["PredType"] = "predict"
            df_score["Top1Class"] = df_score[self.too_class].idxmax(1)
            df_score["Top2Class"] = df_score[self.too_class].apply(lambda x: x.sort_values(ascending=False).index[1], axis=1)
            df_score["Top3Class"] = df_score[self.too_class].apply(lambda x: x.sort_values(ascending=False).index[2], axis=1)
            df_score["AllClass"] = df_score[self.too_class].apply(lambda x: ",".join(x.sort_values(ascending=False).index), axis=1)
            df_score["AllScore"] = df_score[self.too_class].apply(lambda x: ",".join(x.round(4).sort_values(ascending=False).astype(str)), axis=1)
        else:
            raise ValueError("The TOO model does not support algorithms other than mean")

        train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
        df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()
        self._score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
        self._score = self._score.drop_duplicates(subset=["SampleID"], keep="last")
        return self._score

    def varimp(self, method="mean"):
        rslt = []
        for model_id in self.base_models:
            f_model = f"{self.d_base_models}/{model_id}.gsml"
            _model = load_model(f_model, use_predict=True)
            df_t = _model.varimp(method=method)
            rslt.append(df_t)

        df_imp = pd.concat(rslt, ignore_index=True, sort=False)
        if method == "mean":
            df_imp = df_imp.groupby("variable")[["relative_importance", "scaled_importance", "percentage"]].agg(
                "mean").reset_index()

        return df_imp

    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p

