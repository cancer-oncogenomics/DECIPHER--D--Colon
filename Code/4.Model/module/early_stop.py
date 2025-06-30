#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/29 14:51
# @Author  : shenny
# @File    : early_stopp.py
# @Software: PyCharm

"""早停策略"""


class EarlyStopping(object):

    """  模型训练早停策略

    :param patience: int, 早停的容忍次数
    :param mode: str, 早停的模式. [min, max]
    :param min_step: float, 早停的最小步长
    """

    def __init__(self, patience, mode, min_step):
        self.patience = int(patience)
        self.counter = 0
        self.best_value = None
        self.mode = mode
        self.min_step = float(min_step)

        self.early_stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value

        elif self.mode == "min" and value > (self.best_value - self.min_step):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif self.mode == "max" and value < (self.best_value + self.min_step):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = value
            self.counter = 0
        return self.early_stop


class MultiEarlyStopping(object):
    """  多指标早停策略
    :param stop_methods: list, 早停的方法. ["loss,10,min,0.01", "acc,10,max,0.01"]
    :param stopper: list, 早停的方法. [("loss", EarlyStopping), ("acc", EarlyStopping)]
    """

    def __init__(self, stop_methods: list):
        self.stop_methods = stop_methods

        self.stopper = [self.get_stopper(method) for method in stop_methods]

    @staticmethod
    def get_stopper(method):
        """  获取早停实例

        :param method:  str, 早停的方法. "loss,10,min,0.01". loss为指标名, 10为早停的容忍次数, min为早停的模式, 0.01为早停的最小步长
        :return:
        """
        value_tag, patience, mode, min_step = method.split(",")
        stopper = EarlyStopping(int(patience), mode, float(min_step))
        return value_tag, stopper

    @property
    def early_stop(self):
        """ 是否早停

        所有的早停方法都早停了, 则返回True
        :return:
        """

        for _, stopper in self.stopper:
            if not stopper.early_stop:
                return False
        return True

    def __call__(self, value_dict):
        """记录一次指标值，并判断是否早停"""

        for value_tag, stopper in self.stopper:
            stopper(value_dict[value_tag])

