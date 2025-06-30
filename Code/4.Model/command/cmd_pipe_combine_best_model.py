#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/10/10 15:06
# @File     : cmd_pipe_combine_best_model.py
# @Project  : gsml

"""合并最优base model"""

import click

from pipeline.pipe_combine_best_model import PipeCombineBestModel


__all__ = ["cli_pipe_combine_best_model"]


@click.group()
def cli_pipe_combine_best_model():
    pass


@cli_pipe_combine_best_model.command("pipe_combine_best_model")
@click.option("--d_model_list",
              required=True,
              multiple=True,
              help="Path of base model. [cnv,~/test/cnv/]"
              )
@click.option("--d_output",
              required=True,
              help="path of result"
              )
@click.option("--train_info",
              required=True,
              help="Path of train info. [Train,~/test/Train.info.list]"
              )
@click.option("--pred_info",
              multiple=True,
              help="Path of valid info. [Valid,~/test/Valid.info.list]"
              )
@click.option("--feature_list",
              required=True,
              multiple=True,
              help="Path of features."
              )
@click.option("--threads",
              type=click.INT,
              show_default=True,
              help="threads."
              )
@click.option("--n_top_models",
              default="2,3,4,5",
              show_default=True,
              help="n_top_models."
              )
@click.option("--stacked_algo",
              default=["mean", "glm"],
              show_default=True,
              multiple=True,
              help="stacked_algo."
              )
@click.option("--stat_cols",
              show_default=True,
              default="Train_Group,Detail_Group,StageTnm,Sex,ProjectID,Response,SelectGroup,GroupLevel2",
              help="columns name of stat by group"
              )
def cmd_generate_dataset(**kwargs):

    PipeCombineBestModel(**kwargs).select()
