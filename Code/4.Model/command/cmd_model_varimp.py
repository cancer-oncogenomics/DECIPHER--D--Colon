#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/20 13:35

import os
import sys

import click
import h2o

from module.load_model import load_model
from module import cluster

__all__ = ["cli_model_varimp"]


@click.group()
def cli_model_varimp():
    pass


@cli_model_varimp.command("VarImp")
@click.option("--f_model", help="path of gsml model.(One of model,score is a must)")
@click.option("--f_output", required=True, help="Output result path.")
def cmd_model_varimp(f_model, f_output):
    """Output model var"""

    cluster.init()
    model = load_model(f_model, use_predict=True)
    df_imp = model.varimp()
    df_imp.to_csv(f_output, sep="\t", index=False)

    cluster.close()
