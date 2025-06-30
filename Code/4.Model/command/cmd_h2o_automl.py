#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/4 5:15

"""h2o automl训练"""
import os.path

import click

from pipeline.pipe_h2o_automl import pipe_h2o_automl


__all__ = ["cli_h2o_automl"]


@click.group()
def cli_h2o_automl():
    pass


@cli_h2o_automl.command("H2oAutoML")
@click.option("--d_output",
              required=True,
              help="Result output directory"
              )
@click.option("--prefix",
              required=True,
              help="Prefix of output files"
              )
@click.option("--feature",
              required=True,
              multiple=True,
              show_default=True,
              help="Feature file paths for model training and prediction"
              )
@click.option("--train_info",
              required=True,
              multiple=True,
              help="The path to the training info file"
              )
@click.option("--pred_info",
              required=False,
              multiple=True,
              help="The path to the predict info file"
              )
@click.option("--leaderboard",
              multiple=True,
              help="The path to the leaderboard info file"
              )
@click.option("--nthreads",
              type=click.INT,
              default=10,
              show_default=True,
              help="Maximum number of threads used by the H2O service"
              )
@click.option("--max_models",
              type=click.INT,
              default=200,
              show_default=True,
              help="Specify the maximum number of models to build in an AutoML run, excluding the Stacked"
              )
@click.option("--max_runtime_secs_per_model",
              type=click.INT,
              default=1800,
              show_default=True,
              help="Controls the max time the AutoML run will dedicate to each individual model."
              )
@click.option("--max_runtime_secs",
              type=click.INT,
              default=0,
              show_default=True,
              help="Specify the maximum time that the AutoML process will run for"
              )
@click.option("--nfolds",
              type=click.INT,
              default=5,
              show_default=True,
              help="Number of folds for k-fold cross-validation"
              )
@click.option("--seed",
              type=click.INT,
              default=-1,
              show_default=True,
              help="Set a seed for reproducibility"
              )
@click.option("--stopping_metric",
              default="aucpr",
              show_default=True,
              type=click.Choice(["AUTO", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr",
                                 "lift_top_group", "misclassification", "mean_per_class_error", "r2"]),
              help="Specifies the metric to use for early stopping. "
              )
@click.option("--sort_metric",
              default="aucpr",
              show_default=True,
              type=click.Choice(["auc", "aucpr", "logloss", "mean_per_class_error", "rmse", "mse"]),
              help="Metric to sort the leaderboard by at the end of an AutoML run. "
              )
@click.option("--stopping_tolerance",
              type=click.FLOAT,
              default=0.001,
              show_default=True,
              help="Specify the relative tolerance for the metric-based stopping criterion to stop a grid search and "
                   "the training of individual models within the AutoML run. "
              )
@click.option("--weights_column",
              help="The name or index of the column in training_frame that holds per-row weights."
              )
@click.option("--include_algos",
              default=None,
              show_default=True,
              help="List the algorithms to restrict to during the model-building phase. This can’t be used in combination with exclude_algos param."
              )
@click.option("--exclude_algos",
              default=None,
              show_default=True,
              help="List the algorithms to skip during the model-building phase. The full list of options is"
              )
@click.option("--balance_classes",
              is_flag=True,
              help="Specify whether to oversample the minority classes to balance the class distribution. This option can increase"
              )
def cmd_h2o_automl(**kwargs):
    """pipeline of h2o AutoML."""

    if not os.path.exists(kwargs["d_output"]):
        try:
            os.makedirs(kwargs["d_output"])
        except:
            pass

    if kwargs.get("include_algos"):
        kwargs["include_algos"] = kwargs["include_algos"].split(",")
    if kwargs.get("exclude_algos"):
        kwargs["exclude_algos"] = kwargs["exclude_algos"].split(",")

    pipe_h2o_automl(**kwargs)

