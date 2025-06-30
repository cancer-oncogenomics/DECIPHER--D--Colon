#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/31 1:41
# @File     : cmd_h2o_automl_too_two_layer.py
# @Project  : gsml

"""h2o automl训练"""

import click

from pipeline.pipe_h2o_automl_too_two_layer import PipeH2oAutoMlTooTwoLayer


__all__ = ["cli_h2o_automl_too_two_layer"]


@click.group()
def cli_h2o_automl_too_two_layer():
    pass


@cli_h2o_automl_too_two_layer.command("h2o_automl_too_two_layer")
@click.option("--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("--train_info",
              required=True,
              help="train_info"
              )
@click.option("--valid_info",
              required=True,
              help="valid_info"
              )
@click.option("--features",
              multiple=True,
              required=True,
              help="features.like: train,train.csv"
              )
@click.option("--cancer_list",
              required=True,
              help="cancer_list.like:lung,Liver"
              )
@click.option("--cancer_list",
              required=True,
              help="cancer_list.like:lung,live"
              )
@click.option("--nfold",
              type=click.INT,
              default=5,
              show_default=True,
              help="nfold"
              )
@click.option("--nfold_seed",
              type=click.INT,
              default=1,
              show_default=True,
              help="nfold_seed"
              )
@click.option("--force",
              is_flag=True,
              show_default=True,
              help="force"
              )
@click.option("--nthreads",
              type=click.INT,
              default=10,
              show_default=True,
              help="nthreads"
              )
@click.option("--max_models",
              type=click.INT,
              default=200,
              show_default=True,
              help="max_models"
              )
def cli(**kwargs):
    """pipeline of h2o AutoML."""

    kwargs["features"] = {s.split(",")[0]: s.split(",")[1] for s in kwargs["features"]}
    kwargs["cancer_list"] = kwargs["cancer_list"].split(",")

    PipeH2oAutoMlTooTwoLayer(**kwargs).train()

