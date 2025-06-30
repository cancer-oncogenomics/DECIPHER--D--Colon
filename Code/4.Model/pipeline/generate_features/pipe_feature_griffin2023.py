#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 13:37
# @Author  : shenny
# @File    : pipe_feature_griffin2023.py
# @Software: PyCharm

import os
import subprocess


def generate_feature_griffin2023(f_bam, sample_id, f_output, d_tmp=None, threads=5):

    d_data = "/dssg/NGSPipeline/Mercury/Mercury_Databases_20230516/databases/"
    d_tmp = d_tmp or os.path.dirname(f_output)
    os.makedirs(d_tmp, exist_ok=True)

    cmd1 = f"{d_data}/griffin/griffin2023_gc_counts " \
           f"--bam  {f_bam} " \
           f"--bed {d_data}/griffin/hg19.rm_rep.rm_no_mappable.bed " \
           f"--ref_seq {d_data}/griffin/hg19.fa " \
           f"--out {d_tmp}/{sample_id}.GC_counts.txt"

    cmd2 = f"{d_data}/griffin/griffin2023_GC_bias.py " \
           f"--bam_file_name {sample_id} " \
           f"--mapable_name hg19.rm_rep.rm_no_mappable " \
           f"--genome_GC_frequency {d_data}/griffin/hg19_gc_content " \
           f"--out_dir {d_tmp}/ " \
           f"--size_range 15 500"

    cmd3 = f"{d_data}/griffin/griffin_calc_coverage " \
           f"--sample_name {sample_id} " \
           f"--bam {f_bam} " \
           f"--GC_bias {d_tmp}/{sample_id}.GC_bias.txt " \
           f"--sites_yaml {d_data}/griffin/854TF_sites_io_optimize.yaml " \
           f"--results_dir {d_tmp}/ --cpu {threads}"

    cmd4 = f"/dssg/home/sheny/anaconda3/envs/MercuryWeb/bin/python " \
           f"{d_data}/griffin/smooth.py " \
           f"{sample_id} " \
           f"{d_tmp}/{sample_id}.all_sites.coverage.tmp " \
           f"{d_tmp}/{sample_id}.all_sites.coverage.txt"

    cmd5 = f"{d_data}/griffin/griffin2023_combine_854TF_single.py " \
           f"{sample_id} " \
           f"{d_tmp}/{sample_id}.all_sites.coverage.txt {f_output}"

    cmd = " && ".join([cmd1, cmd2, cmd3, cmd4, cmd5])
    subprocess.check_output(cmd, shell=True)
