#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/8/31 14:29
# @File     : cmd_model_select.py
# @Project  : gsml

"""根据模型各项指标，筛选出符合条件的模型"""

import os

import click

from module.model_select import ModelSelect


__all__ = ["cli_model_select"]


@click.group()
def cli_model_select():
    pass


@cli_model_select.command("ModelSelect")
@click.option("--d_model", required=True, help="Path of base model files")
@click.option("--d_model_stat", required=True, help="Path of base model stat files")
@click.option("--f_conf", required=True, help="Profile for model filtering")
@click.option("--f_output", required=False, help="Path of result file")
@click.option("--threads", type=click.INT, default=1, help="Maximum number of threads")
def cli(**kwargs):
    """基模型筛选"""

    ModelSelect(**kwargs).selected()
