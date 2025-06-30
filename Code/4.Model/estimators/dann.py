# -*-coding: Utf-8 -*-
# @File : dann .py
# author: 沈益
# Time：2023/11/25

"""DANN模型的实现"""

from collections import defaultdict
from itertools import cycle
import os
import typing as t

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import ray
from ray import tune
from ray import train as ray_train
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from module.early_stop import MultiEarlyStopping
from module.preprocess import MinMaxScale
from module.frame import TorchFrame
from module import log
from version import __version__


class DANN(nn.Module):
    """DANN模型实例
    使用三个卷积层，两个全连接层，一个领域分类器，一个分类器

    :param input_size: 输入数据的大小

    """

    def __init__(self, input_size: int, num_class: int, num_domain: int,
                 out1: int, conv1: int, pool1: int, drop1: float,
                 out2: int, conv2: int, pool2: int, drop2: float,
                 fc1: int, fc2: int, drop3: float):
        super(DANN, self).__init__()

        input_size = int(input_size)
        num_class = int(num_class)
        num_domain = int(num_domain)
        out1 = int(out1)
        conv1 = int(conv1)
        pool1 = int(pool1)
        drop1 = float(drop1)
        out2 = int(out2)
        conv2 = int(conv2)
        pool2 = int(pool2)
        drop2 = float(drop2)
        fc1 = int(fc1)
        fc2 = int(fc2)
        drop3 = float(drop3)

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out1),  # 规范化激活，将激活值规范到均值为0，方差为1的正态分布
            nn.Dropout(drop1),  # 随机失活，防止过拟合
            nn.MaxPool1d(kernel_size=pool1, stride=2),  # 最大池化，降低维度

            nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2),

            nn.Conv1d(in_channels=out2, out_channels=out2 * 4, kernel_size=conv2, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(out2 * 4),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2)
        )

        self.fc_input_size = self._get_fc_input_size(input_size)

        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, num_class),
        )

        # 领域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, num_domain),
        )

    def forward(self, x, alpha):
        """前向传播
        alpha: 用于控制梯度反转的参数
        """

        features = self.feature_extractor(x)
        features = features.view(-1, self.fc_input_size)  # 将特征展平

        class_output = self.class_classifier(features)  # 分类器

        reverse_features = ReverseLayerF.apply(features, alpha)  # 领域分类器
        domain_output = self.domain_classifier(reverse_features)
        return class_output, domain_output

    def _get_fc_input_size(self, input_size):
        """ 返回特征提取器的输入大小

        :param input_size: 输入数据的大小
        :return: 特征提取器的输入大小
        """

        x = torch.randn(1, 1, input_size)
        x = self.feature_extractor(x)
        return x.shape[1] * x.shape[2]


class ReverseLayerF(torch.autograd.Function):
    """反向传播时，将梯度反转的函数"""

    @staticmethod
    def forward(ctx, x, alpha):
        """前向传播"""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        output = grad_output.neg() * ctx.alpha

        return output, None


class GsDANN(object):

    def __init__(self, verbose=1):

        # 模型训练后，需要存的结果
        self.is_trained = False
        self.d_output = None
        self.model_name = None
        self.framer = None
        self.init_params = None  # 模型初始化参数
        self._score = pd.DataFrame()
        self.metrics = pd.DataFrame()  # 训练过程中的指标

        self.verbose = verbose
        self.gsml_version = __version__  # gsml版本
        self.pytorch_version = torch.__version__  # pytorch版本
        self.algorithm = "Pytorch--DANN"

    def init_framer(self, f_feature, f_train, scale_method, na_strategy):
        """初始化数据处理器"""

        log.info(f"初始化数据处理器", self.verbose)
        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)

        self.framer = TorchFrame()
        self.framer.fit(df_feature, df_train, ["Response", "Domain"], scale_method, na_strategy)
        log.info(f"Response: {self.framer.classes['Response']}", self.verbose)
        log.info(f"Domain: {self.framer.classes['Domain']}", self.verbose)

    def train(self, f_feature, f_train, f_valid, d_output, model_name, init_params: dict,
              lr: float, weight_decay: float, batch_size: int, lambda_domain: float, epochs: int,
              retrain=False, disable_dann: bool = True, early_strategies: str = None
              ):

        assert self.framer, "数据处理器未初始化"

        self.is_trained = self.is_trained if not retrain else False
        self.d_output = self._outdir(d_output)
        self.model_name = model_name
        self.init_params = init_params
        self.init_params["input_size"] = len(self.framer.features)
        self.init_params["num_class"] = len(self.framer.classes["Response"])
        self.init_params["num_domain"] = len(self.framer.classes["Domain"])

        log.info("读取数据", self.verbose)
        df_feature = pd.read_csv(f_feature, low_memory=False)
        df_train = pd.read_csv(f_train, sep="\t", low_memory=False)
        df_valid = pd.read_csv(f_valid, sep="\t", low_memory=False)
        df_train_domain = df_train[~df_train.Domain.isna()]
        df_valid_domain = df_valid[~df_valid.Domain.isna()]

        ds_train = self.framer.create_tensor_dataset(df_feature, df_train, ["Response"])
        domain_train = self.framer.create_tensor_dataset(df_feature, df_train_domain, ["Response", "Domain"])
        ds_valid = self.framer.create_tensor_dataset(df_feature, df_valid, ["Response"])
        domain_valid = self.framer.create_tensor_dataset(df_feature, df_valid_domain, ["Response", "Domain"])

        # 模型训练
        log.info("模型训练", self.verbose)
        f_model_params = self.f_model_params if self.is_trained and not retrain else None
        f_opt_params = self.f_opt_params if self.is_trained and not retrain else None
        model, optimizer, metrics = train_dann(ds_train=ds_train,
                                               ds_valid=ds_valid,
                                               domain_valid=domain_valid,
                                               domain_train=domain_train,
                                               init_params=init_params,
                                               model_state_dict=f_model_params,
                                               opt_state_dict=f_opt_params,
                                               device=self.device,
                                               disable_dann=disable_dann,
                                               early_strategies=early_strategies,
                                               lr=lr,
                                               weight_decay=weight_decay,
                                               batch_size=batch_size,
                                               lambda_domain=lambda_domain,
                                               epochs=epochs,
                                               )

        # 保存预测得分
        log.info("保存预测得分", self.verbose)
        self.is_trained = True
        df_score = self.predict(model, f_feature=f_feature, pred_type="predict")
        df_score.loc[df_score.SampleID.isin(df_train.SampleID), "PredType"] = "train"
        df_score.to_csv(self.f_predict, sep="\t", index=False)
        self._score = df_score.copy()

        # 保存结果
        log.info("保存结果", self.verbose)
        model.to("cpu")
        model.eval()
        torch.save(model, self.f_model)
        torch.save(model.state_dict(), self.f_model_params)
        torch.save(optimizer.state_dict(), self.f_opt_params)
        self.metrics = pd.DataFrame(metrics.data)
        self.metrics.to_csv(self.f_train_process, sep="\t", index=False)

    def predict(self, model=None, f_feature=None, f_output=None, pred_type="predict", save = False):

        assert self.is_trained, "模型未训练"

        model = model or torch.load(self.f_model)
        model = model.to("cpu")
        model.eval()

        df_feature = pd.read_csv(f_feature, low_memory=False)
        X = self.framer.transform_x(df_feature)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.unsqueeze(1).to("cpu")
        class_output, _ = model(X, 1)
        class_output = nn.Softmax(dim=1)(class_output)
        df_score = pd.DataFrame(class_output.detach().numpy(), columns=self.framer.classes["Response"])
        df_score.insert(0, "SampleID", df_feature.SampleID)
        df_score.insert(1, "PredType", pred_type)
        if "Cancer" in self.framer.classes["Response"]:
            df_score.insert(2, "Score", df_score.Cancer)

        if f_output:
            df_score.to_csv(f_output, sep="\t")

        if save:
            if len(self._score):
                train_ids = list(self._score.loc[self._score.PredType == "train", "SampleID"])
            else:
                train_ids = []
            df_out_train = df_score[~df_score.SampleID.isin(train_ids)].copy()  # 去除训练集中的样本
            self._score = pd.concat([self._score, df_out_train], ignore_index=True, sort=False)
            self._score = self._score.drop_duplicates(subset=["SampleID"], keep="last")  # 去除重复的样本, 保留最后一次预测的结果

        return df_score

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def f_model(self):
        """模型实例"""

        return f"{self.d_output}/{self.model_name}.model.pt"

    @property
    def f_predict(self):
        """预测得分文件"""

        return f"{self.d_output}/{self.model_name}.Predict.tsv"

    @property
    def f_model_params(self):
        """模型参数文件"""

        return f"{self.d_output}/{self.model_name}.model.params.pt"

    @property
    def f_opt_params(self):
        """优化器参数文件"""

        return f"{self.d_output}/{self.model_name}.opt.params.pt"

    @property
    def f_train_process(self):
        """训练过程文件"""

        return f"{self.d_output}/{self.model_name}.train.process.tsv"

    @staticmethod
    def _outdir(p):
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
        return p


class Metrics(object):
    """记录模型训练过程中的一些指标

    :param metrics:  需要记录的性能指标，report_metrics时会输出这些指标
    """

    def __init__(self, epoch: int):

        self.metrics = ["loss", "loss_class", "loss_domain", "acc", "recall", "f1"] # 所有的指标
        self.data = []  # 所有epoch的指标，包括metrics和tp,fp,tn,fn等

        # 当前epoch的指标
        self.epoch = epoch
        self.epoch_data = dict()  # {"train": {"tp": [1, 2]}}

    def report_metric(self, precision=4):
        """  返回当前epoch的指标

        :param precision: 小数位数
        """

        return {k: round(v, precision) if k != "epoch" else v for k, v in self.data[-1].items()}

    def stat_epoch(self):
        """统计当前epoch的性能"""

        rslt = {"epoch": self.epoch}
        all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
        for ds, v in self.epoch_data.items():
            tp = sum(v.get("tp", np.nan))
            fp = sum(v.get("fp", np.nan))
            tn = sum(v.get("tn", np.nan))
            fn = sum(v.get("fn", np.nan))
            rslt[f"{ds}__accuracy"] = (tp + tn) / (tp + fp + tn + fn)
            rslt[f"{ds}__recall"] = tp / (tp + fn)
            rslt[f"{ds}__f1"] = 2 * tp / (2 * tp + fp + fn)
            rslt[f"{ds}__loss"] = np.mean(v.get("loss", np.nan))
            rslt[f"{ds}__loss_class"] = np.mean(v.get("loss_class", np.nan))
            rslt[f"{ds}__loss_domain"] = np.mean(v.get("loss_domain", np.nan))

            all_tp += tp
            all_fp += fp
            all_tn += tn
            all_fn += fn

        rslt["accuracy"] = (all_tp + all_tn) / (all_tp + all_fp + all_tn + all_fn)
        rslt["recall"] = all_tp / (all_tp + all_fn)
        rslt["f1"] = 2 * all_tp / (2 * all_tp + all_fp + all_fn)

        self.data.append(rslt)

    def next_epoch(self, epoch):
        """重置当前epoch的指标"""

        self.stat_epoch()
        self.epoch = epoch
        self.epoch_data = dict()

    def __call__(self, value: t.Union[int, float], name: str, dataset: str):
        """记录当前epoch的指标"""

        if dataset not in self.epoch_data:
            self.epoch_data[dataset] = defaultdict(list)
        self.epoch_data[dataset][name].append(value)


def train_dann(ds_train, ds_valid, domain_valid, domain_train, init_params: dict, lr, weight_decay, batch_size,
               lambda_domain, model_state_dict: str = None, opt_state_dict: str = None, device: str = None,
               epochs: int = None, disable_dann: bool = False, early_strategies: str = None, verbose: int = 1):

    # 确认模型,优化器,损失函数
    log.info("初始化模型,优化器,损失函数", verbose)
    model = DANN(**init_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    if model_state_dict:
        log.warning(f"加载模型参数: {model_state_dict}", verbose)
        model.load_state_dict(torch.load(model_state_dict, map_location=device))
    if opt_state_dict:
        log.warning(f"加载优化器参数: {opt_state_dict}", verbose)
        optimizer.load_state_dict(torch.load(opt_state_dict, map_location=device))

    # 数据加载器
    log.info("初始化数据加载器", verbose)
    train_iter = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    valid_iter = DataLoader(ds_valid, batch_size=batch_size, shuffle=True)
    d_train_iter = cycle(DataLoader(domain_train, batch_size=batch_size, shuffle=True))
    d_valid_iter = cycle(DataLoader(domain_valid, batch_size=batch_size, shuffle=True))

    # 初始化早停策略
    log.info("初始化早停策略", verbose)
    early_stopper = MultiEarlyStopping(early_strategies) if early_strategies else None

    # 训练模型
    log.info("训练模型", verbose)
    metrics = Metrics(0)
    for epoch in range(epochs):

        # 分别预测训练集和验证集
        for dataset, data_iter, domain_iter in zip(["train", "valid"], [train_iter, valid_iter], [d_train_iter, d_valid_iter]):

            if dataset == "train":
                model.train()
            else:
                model.eval()

            for i, (X, Y, T) in enumerate(data_iter):

                # 读取并合并domain数据
                X_domain, Y_domain, D, _ = next(domain_iter)
                X = torch.cat([X_domain, X], dim=0)
                Y = torch.cat([Y_domain, Y], dim=0)
                X, Y, D = X.unsqueeze(1).to(device), Y.to(device), D.to(device)

                # 计算alpha
                if dataset == "valid":
                    alpha = 1
                elif not disable_dann:  # train数据集，且指定了alpha
                    p = float(i + epoch * len(train_iter)) / (epochs * len(train_iter))
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    alpha = 0

                # 预测结果
                class_output, domain_output = model(X, alpha)
                loss_class = loss_func(class_output, Y)

                # 计算损失
                domain_output = domain_output[:len(X_domain)]  # domain_output只取D部分
                loss_domain = loss_func(domain_output, D)
                loss = loss_class + loss_domain * lambda_domain

                # 反向传播
                if dataset == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 记录指标
                pos_index = torch.nonzero(T).squeeze()
                neg_index = torch.nonzero(1 - T).squeeze()
                pos_index = pos_index if pos_index.ndim else pos_index.unsqueeze(0)
                neg_index = neg_index if neg_index.ndim else neg_index.unsqueeze(0)
                tp = (class_output[pos_index].argmax(dim=1) == Y[pos_index].argmax(dim=1)).sum().item() if len(pos_index) else 0
                tn = (class_output[neg_index].argmax(dim=1) == Y[neg_index].argmax(dim=1)).sum().item() if len(neg_index) else 0
                fp = len(neg_index) - tn
                fn = len(pos_index) - tp

                metrics(loss.item(), "loss", dataset)
                metrics(loss_class.item(), f"loss_class", dataset)
                metrics(loss_domain.item(), f"loss_domain", dataset)
                metrics(tp, "tp", dataset)
                metrics(fp, "fp", dataset)
                metrics(tn, "tn", dataset)
                metrics(fn, "fn", dataset)

        metrics.next_epoch(epoch + 1)
        print(metrics.report_metric(4))
        ray_train.report(metrics=metrics.report_metric(4))

        # 早停
        if early_strategies:
            # print(metrics.data[-1])
            early_stopper(metrics.data[-1])
            if early_stopper.early_stop:
                break

    return model, optimizer, metrics
