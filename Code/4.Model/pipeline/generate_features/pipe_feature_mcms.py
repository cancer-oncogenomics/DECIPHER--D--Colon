#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 15:10
# @Author  : shenny
# @File    : pipe_feature_mcms.py
# @Software: PyCharm


import os
import subprocess

import pandas as pd


def generate_feature_mcms(f_bam, f_summary, ds_level, sample_id, f_mcms, f_mc=None, d_tmp=None, threads=5):
    """ MC和MCMS特征生成命令

    :param f_bam:
    :param f_summary:
    :param sample_id:
    :param f_mc:
    :param f_mcms:
    :param d_tmp:
    :param threads:
    :param ds_level:
    :return:
    """

    d_soft = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/softwares/"
    d_tmp = d_tmp or os.path.dirname(f_mcms)
    os.makedirs(d_tmp, exist_ok=True)

    # 生成MCMS特征
    cmd1 = f"/dssg/home/hep/.conda/envs/pointy/bin/snakemake " \
           f"--snakefile {d_soft}/MCMS-ILMN-298/scripts/mutsig.smk " \
           f"-d {d_tmp} --config " \
           f"WD={d_soft}/MCMS-ILMN-298 " \
           f"SID={sample_id} " \
           f"BAM={f_bam} " \
           f"BAI={f_bam}.bai " \
           f"QCS={f_summary} " \
           f"MODE={ds_level} " \
           f"--rerun-triggers mtime " \
           f"--cores {threads}"

    cmd2 = f"source ~xuxx/Pipelines/dev/mercury/bashrc && " \
           f"/dssg/home/xuxx/Pipelines/dev/mercury/misc/rename_a_change_mc.py " \
           f"{d_tmp}/{sample_id}.{ds_level}.MCMS.csv " \
           f"{ds_level} " \
           f"{f_mcms}"
    cmd = " && ".join([cmd1, cmd2])
    subprocess.check_output(cmd, shell=True)

    # 生成MC特征
    mc_cols = [
        "SampleID", "MC.ACAA", "MC.ACAC", "MC.ACAG", "MC.ACAT", "MC.ACGA", "MC.ACGC", "MC.ACGG", "MC.ACGT", "MC.ACTA",
        "MC.ACTC", "MC.ACTG", "MC.ACTT", "MC.ATAA", "MC.ATAC", "MC.ATAG", "MC.ATAT", "MC.ATCA", "MC.ATCC", "MC.ATCG",
        "MC.ATCT", "MC.ATGA", "MC.ATGC", "MC.ATGG", "MC.ATGT", "MC.CCAA", "MC.CCAC", "MC.CCAG", "MC.CCAT", "MC.CCGA",
        "MC.CCGC", "MC.CCGG", "MC.CCGT", "MC.CCTA", "MC.CCTC", "MC.CCTG", "MC.CCTT", "MC.CTAA", "MC.CTAC", "MC.CTAG",
        "MC.CTAT", "MC.CTCA", "MC.CTCC", "MC.CTCG", "MC.CTCT", "MC.CTGA", "MC.CTGC", "MC.CTGG", "MC.CTGT", "MC.GCAA",
        "MC.GCAC", "MC.GCAG", "MC.GCAT", "MC.GCGA", "MC.GCGC", "MC.GCGG", "MC.GCGT", "MC.GCTA", "MC.GCTC", "MC.GCTG",
        "MC.GCTT", "MC.GTAA", "MC.GTAC", "MC.GTAG", "MC.GTAT", "MC.GTCA", "MC.GTCC", "MC.GTCG", "MC.GTCT", "MC.GTGA",
        "MC.GTGC", "MC.GTGG", "MC.GTGT", "MC.TCAA", "MC.TCAC", "MC.TCAG", "MC.TCAT", "MC.TCGA", "MC.TCGC", "MC.TCGG",
        "MC.TCGT", "MC.TCTA", "MC.TCTC", "MC.TCTG", "MC.TCTT", "MC.TTAA", "MC.TTAC", "MC.TTAG", "MC.TTAT", "MC.TTCA",
        "MC.TTCC", "MC.TTCG", "MC.TTCT", "MC.TTGA", "MC.TTGC", "MC.TTGG", "MC.TTGT"
    ]
    df_mcms = pd.read_csv(f_mcms)
    df_mcms = df_mcms.rename(columns={c: c.replace("MCMS", "MC") for c in df_mcms.columns if c.startswith("MCMS")})
    df_mc = df_mcms[mc_cols]
    df_mc.to_csv(f_mc, index=False)





