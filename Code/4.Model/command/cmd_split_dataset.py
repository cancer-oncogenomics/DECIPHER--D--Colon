#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 17:54
# @File     : cmd_split_dataset.py
# @Project  : gsml

"""拆分数据集"""

import os

import click

from module.split_dataset import split_dataset


__all__ = ["cli_split_dataset"]


@click.group()
def cli_split_dataset():
    pass


@cli_split_dataset.command("split_dataset")
@click.option("-i", "--dataset",
              required=True,
              multiple=True,
              help="Path to the data set that needs to be split"
              )
@click.option("-t", "--f_train",
              required=True,
              help="output file of train dataset"
              )
@click.option("-e", "--f_test",
              required=True,
              help="output file of test dataset"
              )
@click.option("-s", "--train_size",
              default=0.6,
              type=click.FLOAT,
              show_default=True,
              help=" should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split."
              )
@click.option("-r", "--random_state",
              type=click.INT,
              default=1,
              show_default=True,
              help="Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls."
              )
@click.option("-s", "--stratify",
              help="TIf not None, data is split in a stratified fashion, using this as the class labels. "
              )
def cmd_split_dataset(**kwargs):

    for path in [os.path.dirname(kwargs["f_train"]), os.path.dirname(kwargs["f_test"])]:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                pass

    split_dataset(**kwargs)
