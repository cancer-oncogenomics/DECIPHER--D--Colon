#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 15:38
# @Author  : shenny
# @File    : pipe_feature_motif_end.py
# @Software: PyCharm

import os
import subprocess


def generate_feature_motif_end(f_bam, f_output):
    """ 生成motif end特征
    即motif_end.100-220特征

    :param f_bam:
    :param f_output:
    :return:
    """

    d_data = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/databases/"
    d_soft = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/softwares/"
    d_reposit = "/dssg/NGSPipeline/Mercury/MercuryAnalysisPipline_HQ/"

    cmd = f"source ~xuxx/Pipelines/dev/mercury/bashrc && " \
          f"{d_reposit}/misc/bam_motifend_100-220_noSex.py " \
          f"{d_soft}/bin/sambamba " \
          f"{f_bam} " \
          f"{d_data}/hs37d5.fa 6 > {f_output}.tmp && " \
          f"{d_reposit}/misc/modify_col_name.py " \
          f"-i {f_output}.tmp " \
          f"-t motif.end.100-220 " \
          f"-o {f_output}"
    subprocess.check_output(cmd, shell=True)