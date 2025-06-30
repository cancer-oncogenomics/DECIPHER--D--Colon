#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/20 13:52

import click

from pipeline.predict import predict
from module import cluster


__all__ = ["cli_predict"]


@click.group()
def cli_predict():
    pass


@cli_predict.command("Predict")
@click.option("-i", "--f_model")
@click.option("--feature", multiple=True)
@click.option("--dataset", multiple=True)
@click.option("-n", "--nthreads", type=click.INT, default=10, show_default=True)
@click.option("--max_mem_size", default="20000M", show_default=True)
@click.option("-o", "--f_output", default="")
@click.option("--skip_in_model", is_flag=True, default=False, show_default=True)
@click.option("--submit_shell", is_flag=True, show_default=True)
@click.option("--precise", is_flag=True, show_default=True)
@click.option("--skip_h2o", is_flag=True, show_default=True)
def cmd_predict(nthreads, max_mem_size, skip_h2o, **kwargs):
    if not skip_h2o:
        max_mem_size = f"{nthreads * 2000}M"
        cluster.init(nthreads=nthreads, max_mem_size=max_mem_size)
        predict(**kwargs)
        cluster.close()
    else:
        predict(**kwargs)