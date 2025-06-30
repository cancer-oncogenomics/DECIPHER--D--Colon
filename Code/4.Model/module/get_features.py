#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/8/5 10:58
# @File     : download_features.py
# @Project  : gsml

import logging
import os.path

import coloredlogs
import pandas as pd
import pymongo


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def get_features(id_list: list, ds_level:list=None, local_features:list=None, d_output=None, prefix="Mercury",
                 feature_list:list=None, list_all_features=False):
    """根据样本ID，获得对应的特征文件"""

    logger.info("get features start")
    pyclient = pymongo.MongoClient("mongodb://Mercury:cf5ed606-2e64-11eb-bbe1-00d861479082@10.1.2.171:27001/")
    db = pyclient["Features"]
    db_qc = pyclient["Mercury"]["mercury_qc"]
    depth_dict = {i["SampleID"]: i["MeanDepth"] for i in db_qc.find({})}

    skip_features = [
        "3X_griffin.334TF", "5X_griffin.334TF", "Raw_griffin.334TF",
        "3X_griffin.854TF", "5X_griffin.854TF", "Raw_griffin.854TF",
    ]

    if list_all_features:
        print(db.list_collection_names())
        pyclient.close()
        return

    # rslt = []
    unfind_rslt = []
    for col_name in db.list_collection_names():
        level, n_feature = col_name.split("_", 1)
        low_depth = {"5X": 5.0000001, "3X": 3}.get(level, 0)
        target_ids = [i for i in id_list if depth_dict[i] >= low_depth]

        if level in ds_level and n_feature in feature_list:

            if f"{level}_{n_feature}" in skip_features:
                continue

            logger.info(f"get features: {n_feature} {level}")
            conn = db[col_name]
            data = list(conn.find({"SampleID": {"$in": target_ids}}, {"_id": 0, "md5": 0, "UpdateDate": 0}))
            if not len(data):
                continue

            df_t = pd.DataFrame(data)
            columns = ["SampleID"] + [c for c in df_t.columns if c != "SampleID"]
            df_t = df_t[columns]
            df_t = df_t.rename(columns={c: c.replace("__", ".") for c in df_t.columns})

            if len(df_t) and d_output:
                df_t.to_csv(f"{d_output}/{prefix}.{n_feature}.{level}.csv", index=False)

            if len(df_t) != len(target_ids):
                unfind_ids = set(target_ids) - set(df_t.SampleID)
                for sample_id in unfind_ids:
                    unfind_rslt.append({"SampleID": sample_id, "feature": n_feature, "level": level})
                logger.warning(f"some sample not find in feature <{col_name}>: {unfind_ids}")

    # 本地特征分离
    if local_features:
        for f_feature in local_features:
            n_feature = os.path.basename(f_feature)
            logger.info(f"get features: {n_feature}")
            df_t = pd.read_csv(f_feature)
            df_t = df_t[df_t.SampleID.isin(id_list)]
            # rslt.append(df_t)

            if len(df_t) and d_output:
                df_t.to_csv(f"{d_output}/{prefix}.{n_feature}", index=False)

            if len(df_t) != len(id_list):
                unfind_ids = set(id_list) - set(df_t.SampleID)
                for sample_id in unfind_ids:
                    unfind_rslt.append({"SampleID": sample_id, "feature": n_feature, "level": "-"})
                logger.warning(f"some sample not find in local feature <{n_feature}>: {unfind_ids}")

    df_unfind = pd.DataFrame(unfind_rslt)
    if len(df_unfind):
        df_unfind.to_csv(f"{d_output}/{prefix}.unfind.tsv", sep="\t", index=False)
    pyclient.close()
    logger.info(f"Done!")
    # return rslt


if __name__ == '__main__':
    get_features(["PA190U0198-C5C9H2DXNF1-H001K799Y00D",
                  "PA200S0292-C5C9H2DXNF1-H001Y00DK799",
                  "PA200S0295-C5C9H2DXNF1-H001Y00DK799"],
                 ds_level=["3X", "5X", "Raw"]
                 )


