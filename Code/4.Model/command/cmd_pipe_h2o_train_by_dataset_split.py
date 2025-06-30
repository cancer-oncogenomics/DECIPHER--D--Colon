#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/2 20:29
# @File     : cmd_pipe_h2o_train_by_dataset_split.py
# @Project  : gsml


import click

from pipeline.pipe_h2o_train_by_dataset_split import PipeH2oTrainByDatasetSplit

__all__ = ["cli_pipe_h2o_train_by_dataset_split"]


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli_pipe_h2o_train_by_dataset_split():
    """Command line tool for model training"""

    pass


@cli_pipe_h2o_train_by_dataset_split.command("Pipe_H2oTrainByDatasetSplit", context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file. ps Train.info.list"
              )
@click.option("--pred_info",
              multiple=True,
              help="The path to the predict info file. ps Valid1:Valid1.info.list"
              )
@click.option("--feature",
              required=True,
              multiple=True,
              help="The path to the feature info file. ps cnv:d1.cnv.csv"
              )
@click.option("--prefix",
              required=True,
              show_default=True,
              help="The prefix of the output file"
              )
@click.option("--d_output",
              required=True,
              show_default=True,
              help="The prefix of the output file"
              )
@click.option("--weights_column",
              show_default=True,
              help="The name of the column in training_frame that holds per-row weights."
              )
@click.option("--seed_list",
              default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
              show_default=True,
              help="The seeds of data splitting."
              )
@click.option("--ratio",
              default=0.6,
              type=click.FLOAT,
              show_default=True,
              help="The proportion of Train data set when data is split"
              )
@click.option("--nfolds",
              default=10,
              type=click.INT,
              show_default=True,
              help="nfolds"
              )
@click.option("--algorithms",
              default='glm,gbm,rf,dl,xgboost',
              show_default=True,
              help="algorithms"
              )
@click.option("--fold_assignment",
              default='stratified',
              type=click.Choice(["auto", "random", "modulo", "stratified"]),
              show_default=True,
              help="fold_assignment"
              )
@click.option("--threads",
              default=10,
              type=click.INT,
              show_default=True,
              help="threads"
              )
@click.option("--epochs",
              default=10,
              type=click.INT,
              show_default=True,
              help="epochs"
              )
@click.option("--reproducible",
              default=True,
              is_flag=True,
              show_default=True,
              help="reproducible"
              )
@click.option("--step",
              default="ds_copy,ds_split,base_train,stacked_train,stat,plot",
              show_default=True,
              help="step"
              )
@click.option("--stratify",
              help="TIf not None, data is split in a stratified fashion, using this as the class labels. "
              )
def cmd_pipe_h2o_train_by_dataset_split(**kwargs):
    kwargs["seed_list"] = [int(i) for i in kwargs["seed_list"].split(",")]
    kwargs["algorithms"] = kwargs["algorithms"].split(",")
    kwargs["step"] = kwargs["step"].split(",")

    PipeH2oTrainByDatasetSplit(**kwargs).run()
