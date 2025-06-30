#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/20 13:35

import os
import sys

import click
import yaml

from pipeline.pipe_model_stat import PipeModelStat


__all__ = ["cli_model_stat"]


@click.group()
def cli_model_stat():
    pass


@cli_model_stat.command("ModelStat")
@click.option("--f_model", help="path of gsml model.(One of model,score is a must)")
@click.option("--f_score", help="path of predict score.(One of model,score is a must)")
@click.option("--d_output", required=True, help="Output result path.")
@click.option("--model_name", required=True, help="Output the result file prefix and the result of the Model ID")
@click.option("--dataset", multiple=True, help="Dataset name and path.(ps: Train,train.info.list)")
@click.option("--optimize", multiple=True,  help="Optimize name and path.(ps: KAG9,Optimize_KAG9.tsv)")
@click.option("--cs_conf", help="The profile used by Combine Score")
@click.option("--spec_list",
              multiple=True,
              type=click.FLOAT,
              default=[0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
              show_default=True,
              help="Cutoff was identified using a spectrum and dataset."
              )
@click.option("--sens_list",
              multiple=True,
              type=click.FLOAT,
              show_default=True,
              help="Cutoff was identified using a sens and dataset."
              )
@click.option("--cutoff_dataset", "cutoff_dataset_list",
              multiple=True,
              default=["Train"],
              show_default=True,
              help="Cutoff was identified using a spectrum and dataset."
              )
@click.option("--skip_auc",
              is_flag=True,
              show_default=True,
              default=False,
              help="AUC are not counted"
              )
@click.option("--skip_performance",
              is_flag=True,
              show_default=True,
              default=False,
              help="Performance are not counted"
              )
@click.option("--skip_combine_score",
              is_flag=True,
              show_default=True,
              default=False,
              help="CombineScore are not counted"
              )
@click.option("--skip_by_subgroup",
              is_flag=True,
              show_default=True,
              default=False,
              help="StatByGroup are not counted"
              )
@click.option("--stat_cols",
              show_default=True,
              default="Train_Group,Detail_Group,StageTnm,Sex,ProjectID,Response,SelectGroup,GroupLevel2",
              help="columns name of stat by group"
              )
@click.option("--d_base_models",
              help="path of base models"
              )
@click.option("--out_var_imp",
              help="path of base models"
              )
def cmd_model_stat(**kwargs):

    if kwargs["dataset"]:
        kwargs["dataset"] = {d.split(",")[0]: d.split(",")[1] for d in kwargs["dataset"]}
    if kwargs["optimize"]:
        kwargs["optimize"] = {d.split(",")[0]: d.split(",")[1] for d in kwargs["optimize"]}
    if kwargs["cs_conf"]:
        kwargs["cs_conf"] = yaml.load(open(kwargs["cs_conf"]), Loader=yaml.FullLoader)["arg_combine_score"]
    if kwargs["stat_cols"]:
        kwargs["stat_cols"] = kwargs["stat_cols"].split(",")

    PipeModelStat(**kwargs)()
