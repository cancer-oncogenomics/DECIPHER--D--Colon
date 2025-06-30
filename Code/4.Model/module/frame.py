#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/4 5:13

from functools import reduce
import h2o
import pandas as pd

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import TensorDataset

__all__ = ["GsFrame", "TorchFrame"]


class GsFrame(object):
    """ 将input数据转换成GsML格式的数据"""

    def __init__(self, dataset_list=None, feature_list=None, axis=0):

        self.axis = axis

        self.dataset = self._dataset(dataset_list)
        self.feature = self._feature(feature_list, axis=axis)
        self.data = self._data()

    @property
    def c_dataset(self):
        return list(self.dataset.columns)

    @property
    def c_features(self):

        return [c for c in self.feature.columns if c != "SampleID"]

    @property
    def samples(self):

        return list(self.data["SampleID"])

    @property
    def as_pd(self):
        return self.data.copy()
     
    @property
    def as_h2o(self):
        print(self.c_features)
        #col_types = {c: "float" for c in self.c_features}
        col_types = {'Seq.Platform': 'factor','Score':'float','Seq.TubeType':'factor','Seq.SeparationTimeInterval':'factor','TotalFreezeInterval':'factor','Seq.InGsDNAFreezeTimeInterval':'factor'}
        data = h2o.H2OFrame(self.data.copy(), column_types=col_types)
        return data

    @property
    def as_tensor_by_dann(self):
        """ 将数据转换成tensor格式"""

        df_data = self.data.copy()
        X = torch.sentor(df_data[self.c_features].values, dtype=torch.float32)
        y = torch.sentor(df_data["Response"].values, dtype=torch.float32)
        d = torch.sentor(df_data["Domain"].values, dtype=torch.float32)
        return X, y, d

    @staticmethod
    def _dataset(dataset_list):

        if dataset_list:
            df_dataset = pd.concat([pd.read_csv(f, sep="\t", low_memory=False) for f in dataset_list], ignore_index=True, sort=False)
        else:
            df_dataset = pd.DataFrame(columns=["SampleID", "Response"])
        return df_dataset

    @staticmethod
    def _feature(feature_list, axis):
        if feature_list and axis == 0:
            df_feature = pd.concat([pd.read_csv(f, low_memory=False) for f in feature_list], ignore_index=True, sort=False)
        elif feature_list and axis == 1:
            df_feature = reduce(lambda x, y: pd.merge(x, y, on="SampleID", how="outer"), [pd.read_csv(f) for f in feature_list])
        else:
            df_feature = pd.DataFrame(columns=["SampleID"])
        return df_feature

    def _data(self):
        """合并info和feature结果。作为最原始的数据集"""

        if len(self.feature) and len(self.dataset):
            df_final = pd.merge(self.feature, self.dataset, on="SampleID", how="inner")
        elif len(self.feature):
            df_final = self.feature.copy()
        elif len(self.dataset):
            df_final = self.dataset.copy()
        else:
            df_final = pd.DataFrame()

        return df_final


class TorchFrame(object):
    """将数据转换成pytorch适用的格式"""

    def __init__(self):

        self.is_fit = False  # 是否已经fit过
        self.imputer =  None  # NaN填充实例
        self.scaler = None  # 数据缩放实例
        self.features = None  # 所有特征名的集合
        self.classes = dict()  # 所有类别名的集合. {"Response": ["Cancer", "Healthy"], "Domain": ["D1", "D2"]...}

    def fit(self, df_feature: pd.DataFrame, df_dataset: pd.DataFrame = None, class_cols: list = None,
            scale_method: list = None, na_strategy: list = None):
        """ fit数据，获取特征名和类别名

        :param df_feature: train数据集的特征，用于fit缩放和填充实例
        :param df_dataset: train数据集信息，用于获取类别名
        :param class_cols: 需要存储类别信息的类。一般为Response，DANN模型需要Domain列
        :param scale_method: 特征缩放方法。 [minmax]
        :param na_strategy: na填充策略。 [mean]
        :return:
        """

        # 特征及特征处理实例fit
        self.features = [c for c in df_feature.columns if c != "SampleID"]
        if scale_method and na_strategy:
            self.imputer = SimpleImputer(strategy=na_strategy)
            self.scaler = self._get_scaler(scale_method)
            self.imputer.fit(df_feature[self.features])
            self.scaler.fit(self.imputer.transform(df_feature[self.features]))
        elif na_strategy:
            self.imputer = SimpleImputer(strategy=na_strategy)
            self.imputer.fit(df_feature[self.features])
        elif scale_method:
            self.scaler = self._get_scaler(scale_method)
            self.scaler.fit(df_feature[self.features])

        # 类别名
        if class_cols:
            for col in class_cols:
                self.classes[col] = df_dataset[col].unique().tolist()

        self.is_fit = True

    def transform_x(self, df_feature: pd.DataFrame):
        """特征值转换。填充缩放"""

        assert self.is_fit, "fit方法未执行"

        if self.imputer and self.scaler:
            X = self.scaler.transform(self.imputer.transform(df_feature[self.features]))
        elif self.imputer:
            X = self.imputer.transform(df_feature[self.features])
        elif self.scaler:
            X = self.scaler.transform(df_feature[self.features])
        else:
            X = df_feature[self.features]
        return X

    def transform_y(self, df_dataset: pd.DataFrame, class_cols: list):
        """ 类别值转换。one-hot编码

        :param df_dataset: 数据集
        :param class_cols: 待转换的列名
        :return:
        """

        assert self.is_fit, "fit方法未执行"

        rslt = []
        for col in class_cols:
            assert col in df_dataset.columns, f"{col} not in df_dataset"
            assert col in self.classes, f"{col} not in self.classes. {self.classes.keys()}"
            assert set(df_dataset[col].unique()) == set(self.classes[col]), f"{col}类别数不一致。{self.classes[col]}"

            rslt.append(pd.get_dummies(df_dataset[col]).astype(int))

        return rslt

    def create_tensor_dataset(self, df_feature: pd.DataFrame, df_dataset: pd.DataFrame, class_cols: list):
        """将数据转换成tensor格式"""

        df_data = pd.merge(df_feature, df_dataset, on="SampleID", how="inner")
        df_data["Positive"] = df_data["Response"].apply(lambda x: 1 if x == "Cancer" else 0)  # 记录阴阳性

        X = self.transform_x(df_data[self.features])
        X = torch.tensor(X, dtype=torch.float32)

        Y_list = self.transform_y(df_data, class_cols)
        Y_list = [torch.tensor(y.values, dtype=torch.float32) for y in Y_list]
        Y_list.append(torch.tensor(df_data["Positive"].values, dtype=torch.float32))

        dataset = TensorDataset(X, *Y_list)
        return dataset

    @staticmethod
    def _get_scaler(scale_method):
        if scale_method == "minmax":
            return preprocessing.MinMaxScaler()
        else:
            raise ValueError(f"scale_method must be in ['minmax']")
