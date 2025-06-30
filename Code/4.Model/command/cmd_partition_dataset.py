#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 17:54
# @File     : cmd_split_dataset.py
# @Project  : gsml

"""拆分数据集"""

import click

from module.partition_dataset import PartitionDataset


__all__ = ["cli_partition_dataset"]


@click.group()
def cli_partition_dataset():
    pass


@cli_partition_dataset.command("partition_dataset")
@click.option("-f", "--f_conf",
              required=True,
              help="Path to config file"
              )
@click.option("-o", "--d_output",
              help="path of result"
              )
@click.option("-p", "--prefix",
              default="Mercury",
              show_default=True,
              help="prefix"
              )
@click.option("-i", "--inherit",
              help="inherit dataset"
              )
@click.option("--paired_mode",
              is_flag=True,
              help="Enable platform pairing mode"
              )
def cmd_generate_dataset(**kwargs):

    PartitionDataset(**kwargs).partitioning()
