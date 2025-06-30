#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 15:48
# @Author  : shenny
# @File    : cmd_generate_features.py
# @Software: PyCharm

"""用于生成各类mercury特征文件"""


import os

import click

from pipeline.generate_features import *


@click.group()
def cli_generate_features():
    pass


@cli_generate_features.command("gf_cnv")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
def cmd_gf_cnv(**kwargs):
    """生成cnv特征"""

    generate_feature_cnv(**kwargs)


@cli_generate_features.command("gf_frag2023")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_frag", "f_frag",
              required=True,
              help="path of result file"
              )
@click.option("-O", "--f_frag_arm", "f_frag_arm",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
@click.option("-t", "--threads", "threads",
              default=10,
              show_default=10,
              help="max cpus"
              )
def cmd_gf_frag2023(**kwargs):
    """生成frag2023和frag_arm2023特征"""

    generate_feature_frag2023(**kwargs)


@cli_generate_features.command("gf_fragma")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
def cmd_gf_fragma(**kwargs):
    """生成fragma特征"""

    generate_feature_fragma(**kwargs)


@cli_generate_features.command("gf_griffin2023")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
@click.option("-t", "--threads", "threads",
              default=10,
              show_default=10,
              help="max cpus"
              )
def cmd_gf_griffin2023(**kwargs):
    """生成griffin2023.854TF特征"""

    generate_feature_griffin2023(**kwargs)


@cli_generate_features.command("gf_mcms")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-m", "--f_summary", "f_summary",
              required=True,
              help="path of qc file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_mcms", "f_mcms",
              required=True,
              help="path of result file"
              )
@click.option("-O", "--f_mc", "f_mc",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
@click.option("-t", "--threads", "threads",
              default=10,
              show_default=10,
              help="max cpus"
              )
@click.option("-l", "--ds_level", "ds_level",
              required=True,
              help="down sample_level"
              )
def cmd_gf_mcms(**kwargs):
    """生成MCMS和MC特征特征"""

    generate_feature_mcms(**kwargs)


@cli_generate_features.command("gf_mcms_mgi")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-m", "--f_summary", "f_summary",
              required=True,
              help="path of qc file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_mcms", "f_mcms",
              required=True,
              help="path of result file"
              )
@click.option("-O", "--f_mc", "f_mc",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
@click.option("-t", "--threads", "threads",
              default=10,
              show_default=10,
              help="max cpus"
              )
@click.option("-l", "--ds_level", "ds_level",
              required=True,
              help="down sample_level"
              )
def cmd_gf_mcms(**kwargs):
    """生成MCMS和MC特征特征"""

    generate_feature_mcms_mgi(**kwargs)


@cli_generate_features.command("gf_motif_end")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
def cmd_gf_motif_end(**kwargs):
    """motif_end.100-220特征"""

    generate_feature_motif_end(**kwargs)


@cli_generate_features.command("gf_motif_extend")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
def cmd_gf_motif_extend(**kwargs):
    """motif_breakpoint.100-220.csv特征"""

    generate_feature_motif_extend(**kwargs)


@cli_generate_features.command("gf_ocf_tcell")
@click.option("-i", "--f_bam", "f_bam",
              required=True,
              help="path of bam file"
              )
@click.option("-s", "--sample_id", "sample_id",
              required=True,
              help="sample_id"
              )
@click.option("-o", "--f_output", "f_output",
              required=True,
              help="path of result file"
              )
@click.option("-d", "--d_tmp", "d_tmp",
              required=False,
              help="path of tmp dir"
              )
def cmd_gf_ocf_tcell(**kwargs):
    """生成OCF_Tcell特征"""

    generate_feature_ocf_tcell(**kwargs)
