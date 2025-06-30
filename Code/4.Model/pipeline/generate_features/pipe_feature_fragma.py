#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 15:26
# @Author  : shenny
# @File    : pipe_feature_fragma.py
# @Software: PyCharm

import os
import subprocess


def generate_feature_fragma(f_bam, sample_id, f_output, d_tmp=None):
    """ 生成fragma特征

    :param f_bam:
    :param sample_id:
    :param f_output:
    :param d_tmp:
    :return:
    """

    d_data = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/databases/"
    d_reposit = "/dssg/NGSPipeline/Mercury/MercuryAnalysisPipline_HQ_dev/"
    d_tmp = d_tmp or os.path.dirname(f_output)
    os.makedirs(d_tmp, exist_ok=True)

    cmd1 = f'/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python {d_reposit}/misc/FragMa/7.seri2CG.3mer.motif.wgs.py ' \
           f'{d_data}/FragMa/hg19.WG.Seri2Csite.11ntwin.bed ' \
           f'{f_bam} ' \
           f'{d_data}/FragMa/hg19.fa ' \
           f'{d_tmp} {sample_id}'

    cmd2 = f'/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python {d_reposit}/misc/FragMa/4.cgn_ncg.3mer.motif.wgs.py ' \
           f'{d_data}/FragMa/hg19.Alu.Csite.11ntwin.bed ' \
           f'{f_bam} ' \
           f'{d_data}/FragMa/hg19.fa ' \
           f'{d_tmp} {sample_id}'

    cmd3 = f'/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python {d_reposit}/misc/FragMa/fragma.py ' \
           f'{d_tmp}/{sample_id}.1CG.csv ' \
           f'{d_tmp}/{sample_id}.2CG.csv ' \
           f'{f_output}'

    cmd = " && ".join([cmd1, cmd2, cmd3])
    subprocess.check_output(cmd, shell=True)
