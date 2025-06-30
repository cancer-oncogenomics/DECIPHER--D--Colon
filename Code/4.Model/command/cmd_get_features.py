#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/8/5 11:27
# @File     : cmd_get_features.py
# @Project  : gsml


#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 17:54
# @File     : cmd_split_dataset.py
# @Project  : gsml

"""拆分数据集"""

import os

import click

from module.get_features import get_features


__all__ = ["cli_get_features"]


@click.group()
def cli_get_features():
    pass


@cli_get_features.command("get_features")
@click.option("-i", "--id_list", required=False, help="id list file")
@click.option("-l", "--ds_level", multiple=True, default=["5X", "Raw", "LE5X", "tmp"], show_default=True, help="down sample level")
@click.option("--local_features", multiple=True, help="local features")
@click.option("-o", "--d_output", required=False, help="path of output")
@click.option("-p", "--prefix", default="Mercury", show_default=True, help="prefix")
@click.option("--feature_list",
              multiple=True,
              default=["cnv", "fragment_ScaleShortLongPeak1", "fragment_arm.5bp.100-220", "griffin.334TF", "griffin.854TF",
                       "neomer.genegroup", "combine2"],
              show_default=True,
              help="list_all_features"
              )
@click.option("--list_all_features", is_flag=True, show_default=True, help="list_all_features")
def cli(**kwargs):
    """拆分数据集"""

    if kwargs["list_all_features"]:
        get_features(**kwargs)
    else:
        kwargs["id_list"] = [line.strip() for line in open(kwargs["id_list"])]

        if not os.path.exists(kwargs["d_output"]):
            try:
                os.makedirs(kwargs["d_output"])
            except:
                pass

        get_features(**kwargs)
