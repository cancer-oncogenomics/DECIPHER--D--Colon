#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/23 15:58
# @Author  : shenny
# @File    : cmd_search_bam.py
# @Software: PyCharm


import os

import click

from pipeline.pipe_search_bam import pipe_search_bam

__all__ = ["cli_search_bam"]


@click.group()
def cli_search_bam():
    """查询bam文件路径，并链接到指定目录"""
    pass


@cli_search_bam.command("search_bam")
@click.option("-s", "--sample_id",
              help="sample id"
              )
@click.option("-f", "--f_ids",
              help="id list"
              )
@click.option("-l", "--level",
              default="LE5X",
              show_default=True,
              type=click.Choice(["LE5X", "3X", "5X", "Raw"]),
              help="down sample level"
              )
@click.option("-o", "--d_output",
              help="output path. if not specified, will only print search result"
              )
@click.option("--fuzzy",
              is_flag=True,
              show_default=True,
              help="fuzzy search"
              )
@click.option("--qc",
              is_flag=True,
              show_default=True,
              help="find qc summary file"
              )
def cmd_search_bam(**kwargs):
    pipe_search_bam(**kwargs)
