#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/23 15:45
# @Author  : shenny
# @File    : pipe_search_bam.py
# @Software: PyCharm
import os

import pymongo
import subprocess


def pipe_search_bam(level="LE5X", sample_id=None, f_ids=None, qc=False, fuzzy=False, d_output=None):
    """ 查询样本bam文件路径

    :param fuzzy:
    :param sample_id:
    :param f_ids:
    :param level:
    :param qc:
    :param bam:
    :param d_output:
    :return:
    """

    if d_output and not os.path.exists(d_output):
        os.makedirs(d_output, exist_ok=True)

    db = pymongo.MongoClient("mongodb://Mercury:cf5ed606-2e64-11eb-bbe1-00d861479082@10.1.2.171:27001/")
    conn = db["Mercury"]["mercury_sample_path"]

    assert sample_id or f_ids, f"One of the sample_id and f_ids must be specified"

    id_list = [sample_id] if sample_id else [f.strip() for f in open(f_ids)]
    for sample_id in id_list:
        sql = {"SampleID": sample_id} if not fuzzy else {"SampleID": {"$regex": f".*{sample_id}.*"}}
        for data in conn.find(sql):
            f_bam = data.get(f"DedupBamR{level.replace('Raw', 'aw')}", "")
            f_summary = data.get("Summary", "")

            if qc:
                print(f"{sample_id}\t{f_bam}\t{f_summary}")
            else:
                print(f"{sample_id}\t{f_bam}")

            if d_output:
                subprocess.check_output(f"ln -sf {f_bam}* {d_output}/", shell=True)
                if qc:
                    subprocess.check_output(f"ln -sf {f_summary} {d_output}/", shell=True)
