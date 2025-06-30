#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 11:00
# @Author  : shenny
# @File    : pipe_feature_cnv.py
# @Software: PyCharm

import os
import subprocess

import pandas as pd


def generate_feature_cnv(f_bam, sample_id, f_output, d_tmp=None):
    """ cnv特征生成

    :param f_bam: dedup bam文件路径
    :param sample_id: 样本id
    :param f_output: 结果文件路径
    :param d_tmp: 临时文件存放路径
    :return: None
    """

    d_data = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/databases/"
    d_soft = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/softwares/"
    d_reposit = "/dssg/NGSPipeline/Mercury/MercuryAnalysisPipline_HQ/"
    d_tmp = d_tmp or os.path.dirname(f_output)
    os.makedirs(d_tmp, exist_ok=True)

    # 统计1M bin下的 reads count数量
    cmd1 = f"{d_soft}/hmmcopy_utils/bin/readCounter " \
           f"--window 1000000 " \
           f"--quality 20 " \
           f"--chromosome 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 {f_bam} > {d_tmp}/{sample_id}.wig && " \
           f"sed -i \'s/chrom=chr/chrom=/g\' {d_tmp}/{sample_id}.wig"

    # 计算cnv
    cmd2 = f"/dssg/home/xuxx/anaconda3/envs/Mercury/bin/Rscript {d_reposit}/misc/runIchorCNA.R " \
           f"--id {sample_id} " \
           f"--WIG {d_tmp}/{sample_id}.wig " \
           f"--ploidy 'c(2)' " \
           f"--normal 'c(0.95, 0.99, 0.995, 0.999)' " \
           f"--maxCN 3 " \
           f"--gcWig {d_data}/gc_hg19_1000kb.wig " \
           f"--mapWig {d_data}/map_hg19_1000kb.wig " \
           f"--centromere {d_data}/GRCh37.p13_centromere_UCSC-gapTable.txt " \
           f"--normalPanel {d_data}/HD_ULP_PoN_1Mb_median_normAutosome_mapScoreFiltered_median.rds " \
           f"--includeHOMD False " \
           f"--chrs 'c(1:22)' " \
           f"--chrTrain 'c(1:22)' " \
           f"--estimateNormal True " \
           f"--estimatePloidy True " \
           f"--estimateScPrevalence FALSE " \
           f"--scStates 'c()' " \
           f"--txnE 0.9999 " \
           f"--txnStrength 10000 " \
           f"--libdir {d_soft}/ichorCNA " \
           f"--outDir {d_tmp}"

    # 修改文件格式
    cmd3 = f"source ~xuxx/Pipelines/dev/mercury/bashrc && " \
           f"{d_reposit}/misc/modify_col_name.py " \
           f"-i {d_tmp}/{sample_id}.correctedDepth.txt " \
           f"-t cnv " \
           f"-o {f_output}"

    cmd = " && ".join([cmd1, cmd2, cmd3])
    subprocess.check_output(cmd, shell=True)


