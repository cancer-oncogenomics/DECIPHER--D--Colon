#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 13:45
# @Author  : shenny
# @File    : pipe_feature_frag2023.py
# @Software: PyCharm

import os
import subprocess


def generate_feature_frag2023(f_bam, sample_id, f_frag, f_frag_arm, d_tmp=None, threads=5):
    """ frag和frag arm两个特征的命令

    :param f_frag_arm:
    :param f_frag:
    :param f_bam:
    :param sample_id:
    :param d_tmp:
    :param threads:
    :return:
    """

    d_data = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/databases/"
    d_reposit = "/dssg/NGSPipeline/Mercury/MercuryAnalysisPipline_HQ/"
    d_tmp = d_tmp or os.path.dirname(f_frag)
    os.makedirs(d_tmp, exist_ok=True)

    # 生成frag2023特征
    cmd1 = f"/dssg/home/hep/.conda/envs/bioinfo/bin/python3.10 {d_reposit}/misc/frag_arm_2023.py " \
           f"{f_bam} " \
           f"{sample_id} " \
           f"{d_tmp} " \
           f"{threads} && " \
           f"mv {d_tmp}/{sample_id}.fragment_arm.5bp.100-220.csv {f_frag_arm}"

    cmd2 = f"/dssg/home/xuxx/anaconda3/envs/frag/bin/Rscript {d_reposit}/misc/new_frag_wgs_loess_fix.r " \
           f"-s {sample_id} " \
           f"-o {d_tmp} " \
           f"--range_start=100 " \
           f"--short_end=150 " \
           f"--ab {d_data}/AB.rds"

    cmd3 = f"source ~xuxx/Pipelines/dev/mercury/bashrc && " \
           f"{d_reposit}/misc/modify_col_name.py " \
           f"-i {d_tmp}/{sample_id}_fragment_ScaleShortLongPeak1.csv " \
           f"-t frag2023 " \
           f"-o {f_frag}"

    cmd = " && ".join([cmd1, cmd2, cmd3])
    subprocess.check_output(cmd, shell=True)