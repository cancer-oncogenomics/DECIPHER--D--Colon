#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 15:32
# @Author  : shenny
# @File    : pipe_feature_ocf_tcell.py
# @Software: PyCharm

import os
import subprocess


def generate_feature_ocf_tcell(f_bam, sample_id, f_output, d_tmp=None):
    """ ocf_tcell特征生成

    :param f_bam:
    :param sample_id:
    :param f_output:
    :param d_tmp:
    :return:
    """

    d_reposit = "/dssg/NGSPipeline/Mercury/MercuryAnalysisPipline_HQ_dev/"
    d_tmp = d_tmp or os.path.dirname(f_output)
    os.makedirs(d_tmp, exist_ok=True)

    cmd1 = f"/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python {d_reposit}/misc/ocf_tcell/bam2bed.pysam.py " \
           f"{f_bam} " \
           f"{d_tmp} {sample_id}"

    cmd2 = f"/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python {d_reposit}/misc/ocf_tcell/calu_ocf.py" \
           f" {sample_id} {d_tmp} {d_tmp} {d_tmp}"

    if f"{d_tmp}/{sample_id}.ocf.csv" != f_output:
        cmd2 += f" && mv {d_tmp}/{sample_id}.ocf.csv {f_output}"

    cmd = " && ".join([cmd1, cmd2])
    subprocess.check_output(cmd, shell=True)
