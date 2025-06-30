#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/16 20:58
# @File     : pipe_split_dataset.py
# @Project  : gsml

from itertools import combinations
import logging
import os
import random
import subprocess

import pandas as pd
import pymongo
import yaml

import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


__all__ = ["PartitionDataset"]


class PartitionDataset(object):
    """ 数据集拆分

    :param paired_mode: 开启配对模式，config文件支持paired_mode语法


    """

    def __init__(self, f_conf, d_output, prefix="Mercury", inherit=None, paired_mode=False):
        self.f_conf = f_conf
        self.d_output = self.outdir(d_output)
        self.prefix = prefix
        self.d_info = self.outdir(f"{d_output}/info")
        self.df_inherit = pd.read_csv(inherit, sep="\t") if inherit else pd.DataFrame()
        self.paired_mode = paired_mode

        self.client = pymongo.MongoClient('mongodb://Mercury:cf5ed606-2e64-11eb-bbe1-00d861479082@10.1.2.171:27001/')
        self.db = self.client["Mercury"]
        self.collection = self.db["mercury_final"]
        self.col_desc = self.db["mercury_project"]

        self.conf = yaml.load(open(f_conf), Loader=yaml.FullLoader)
        # self.description = dict(self.get_desc(), **self.conf.get("description", {}))
        # self.dataset = self.conf.get("dataset", {})

    def search(self,
               df_all=pd.DataFrame(),
               exclude_platform=False,
               paired_mode=None,
               ratio: list = None,
               seed: int = 0,
               exclude_ids: list = None,
               **sql):
        """ 从数据库查询所需数据，并返回

        1. 根据全局过滤条件，筛选出全集数据集
        2. 根据本条过滤条件，筛选出本次的数据集，
        3. 两者取交集

        :param exclude_platform:
        :param exclude_ids:
        :param seed:
        :param ratio:
        :param paired_mode:
        :param df_all: 全集数据集，筛选出来的子集，需要在全集中
        :param sql:
        :return:
        """

        # 相关参数提取
        raw_sql = sql.copy()
        if paired_mode:
            raw_sql["paired_mode"] = paired_mode
        platform = sql.pop("Platforms") if sql.get("Platforms") and self.paired_mode and exclude_platform else None

        # 错误检查
        if not self.paired_mode and paired_mode:
            raise ValueError("If you want to use pairing mode, specify parameters --paired_mode")
        assert not ("_exclude_file" in sql.keys() and "_include_file" in sql.keys()), f"_exclude_file and _include_file and not in a same sql"

        # 拉取子集
        if "_exclude_file" in sql.keys():
            sql["SampleID"] = {"$nin": [line.strip() for line in open(sql["_exclude_file"])]}
            sql.pop("_exclude_file")
        elif "_include_file" in sql.keys():
            sql["SampleID"] = {"$in": [line.strip() for line in open(sql["_include_file"])]}
            sql.pop("_include_file")

        sql = {k: v for k, v in sql.items()}
        projection = {k: False for k in ["_id", "id", "CreateDate", "UpdateDate", "UploadFile", "UpdateUser"]}
        data = self.collection.find(sql, projection=projection)
        df_data = pd.DataFrame(data)
        if exclude_ids and len(df_data):
            df_data = df_data[~df_data.SampleID.isin(exclude_ids)]

        # 与全集做交集
        if len(df_all) and len(df_data):
            df_data = df_data[df_data.SampleID.isin(df_all.SampleID)]

        # 多平台混合，配对拉取模式
        if not self.paired_mode and paired_mode:
            raise ValueError("If you want to use pairing mode, specify parameters --paired_mode")

        if self.paired_mode and len(df_data):
            if paired_mode:
                df_data = self.get_paired_mode(df_data, paired_mode)

            if platform and exclude_platform:
                if type(platform) == str:
                    df_data = df_data[df_data.Platforms == platform]
                elif type(platform) == dict:
                    if platform.get("$in"):
                        df_data = df_data[df_data.Platforms.isin(platform.values())]
                    elif platform.get("$nin"):
                        df_data = df_data[~df_data.Platforms.isin(platform.values())]

        # 按比例拆分
        if ratio and len(df_data) and paired_mode and paired_mode != "single":
            df_data["shortID"] = df_data["SampleID"].apply(lambda x: x.split("-")[0])
            short_ids = df_data["shortID"].unique().tolist()
            random.seed(1)
            random.shuffle(short_ids)

            total = len(short_ids)
            start = round(total * abs(ratio[0]))
            end = round(total * abs(ratio[1]))
            short_ids = short_ids[start: end]
            df_data = df_data[df_data["shortID"].isin(short_ids)]
            df_data.pop("shortID")

        elif ratio and len(df_data):
            df_data = df_data.sample(frac=1, random_state=seed)
            total = len(df_data)
            start = round(total * abs(ratio[0]))
            end = round(total * abs(ratio[1]))
            df_data = df_data.iloc[start: end]

        # 检查数据集是否为空
        if not len(df_data):
            logger.warning(f"sub set not find samples. {raw_sql}")
            df_data = pd.DataFrame(columns=["SampleID"])
        return df_data

    def partitioning(self):
        """划分数据集"""

        logger.info(f"parse config file")
        subprocess.check_call(f"cp {self.f_conf} {self.d_output}/{self.prefix}.yaml", shell=True)

        # 确定全局过滤条件
        base_sql = self.conf.get("filter", {})

        # 根据基本过滤条件，得到应该可用的所有样本信息
        if base_sql:
            f_cp = base_sql.pop("FilterCancerType") if "FilterCancerType" in base_sql.keys() else None
            df_all = self.search(**base_sql).fillna("")
            if f_cp:
                df_cp = pd.read_excel(f_cp).fillna("")
                df_cp.Train = df_cp.Train.astype(int)
                df_cp = df_cp[df_cp.Train == 1][["Cancer", "Histology"]].rename(columns={"Cancer": "CancerType"})
                df_all = pd.merge(df_all, df_cp, on=["CancerType", "Histology"], how="inner")
            df_all.to_csv(f"{self.d_output}/{self.prefix}.select.Filter.tsv", sep="\t", index=False)
        else:
            df_all = pd.DataFrame(columns=["SampleID"])

        # 根据filter_2信息，确定预排除的样本。并且记录其排除原因
        if self.conf.get("filter_2"):
            rslt = []
            for k, v in self.conf["filter_2"].items():
                for sql in v:
                    df_t = self.search(**sql).fillna("")
                    df_t["ExcludeGroup"] = k
                    rslt.append(df_t)
            df_filter_2 = pd.concat(rslt, ignore_index=True, sort=False)
            df_all = df_all[~df_all.SampleID.isin(df_filter_2.SampleID)].copy()
        else:
            df_filter_2 = pd.DataFrame(columns=["SampleID"])

        # 根据排除条件，得到应该排除的所有样本信息
        if self.conf.get("exclude"):
            rslt = []
            for k, v in self.conf["exclude"].items():
                for sql in v:
                    df_t = self.search(**sql).fillna("")
                    df_t["ExcludeGroup"] = k
                    rslt.append(df_t)
            df_exclude = pd.concat(rslt, ignore_index=True, sort=False)
            df_exclude = df_exclude[df_exclude.SampleID.isin(df_all.SampleID)]
            df_exclude = pd.concat([df_filter_2, df_exclude], ignore_index=True, sort=False)
            df_exclude.to_csv(f"{self.d_output}/{self.prefix}.select.Exclude.tsv", sep="\t", index=False)
        else:
            df_exclude = pd.DataFrame(columns=["SampleID"])

        # 从数据库重新拉取继承数据集的样本信息
        if len(self.df_inherit):
            df_inherit = self.search(**{"SampleID": {"$in": list(self.df_inherit.SampleID)}})
            df_inherit = pd.merge(self.df_inherit, df_inherit, on="SampleID", how="inner")
            df_inherit.to_csv(f"{self.d_output}/{self.prefix}.inherit.tsv", sep="\t", index=False)
        else:
            df_inherit = pd.DataFrame()

        # 根据config,拉取数据集待纳入的样本
        # logger.info(f"select by database")
        samples = []  # 每个子集对应的样本ID，用来确认各个子集是否重复
        rslt = []
        for ds_name, sql_dict in self.conf["dataset"].items():
            for group, sql_list in sql_dict.items():
                for i, sql in enumerate(sql_list):
                    # df_t = self.search(**dict(base_sql, **sql))
                    df_t = self.search(df_all, exclude_platform=True, **sql)
                    if df_t.shape[0]:
                        df_t["Dataset"] = ds_name
                        df_t["SelectGroup"] = group
                        df_t["Describe"] = df_t.ProjectID.apply(lambda x: self.get_desc.get(x))
                        df_t["SelectNote"] = self.conf["select_note"].get(group, "")
                        rslt.append(df_t)
                        samples.append((f"{ds_name}-{group}-{i}", set(df_t.SampleID)))
        df_info = pd.concat(rslt, ignore_index=True, sort=False)
        df_info = df_info[df_info.SampleID.isin(df_all.SampleID)]

        # 3. 确定各个子集是否有重复
        for (name1, set1), (name2, set2) in combinations(samples, 2):
            multi_ids = set1 & set2
            if multi_ids:
                logger.warning(f"There are duplicate samples between the {name1},{name2}: <{multi_ids}>")

        # 4. 确定跟exclude有没有重复
        df_both = df_info[df_info.SampleID.isin(df_exclude.SampleID)]
        if df_both.shape[0]:
            df_both.to_csv(f"{self.d_output}/{self.prefix}.select.bothExclude.tsv", sep="\t", index=False)
            logger.warning(f"There are duplicate samples between include and exclude {';'.join(df_both.SampleID)}")

        # 4. 合并继承的数据集，去重后，数据集文件
        if df_inherit.shape[0]:
            df_info = pd.concat([df_inherit, df_info], ignore_index=True, sort=False)
        df_info = df_info.drop_duplicates(subset="SampleID", keep="first")

        # 5. 调整相应数据，得到最终结果
        df_info["Train_Group"] = df_info["GroupLevel1"]
        df_info["Detail_Group"] = df_info.apply(lambda x: f"{x.GroupLevel2}-{x.GroupLevel4}", axis=1)
        df_info["Response"] = df_info.apply(self.get_response, axis=1)
        report_cols = [
            "SampleID", "Response",  "Train_Group", "Detail_Group",  "ProjectID", "Name", "Age", "Sex", "StageSystem", "Stage",
            "StageTnm", "Group", "TreatmentBeforeSurgery", "TimePoint", "TubeType", "TubeBrand", "ExtractSop",
            "LibrarySop", "SampleGroup", "GroupLevel1", "GroupLevel2", "GroupLevel3", "GroupLevel4", "GroupLevel5", "GroupLevel6",
            "Dataset", "SelectGroup", "HospitalID", "AnalyzedSex", "Platforms", "DateOfCollection", "ReceptionSampleType",
            "DateOfSeparation", "SeparationTimeInterval", "ExtractionTimeInterval", "InHospitalFreezeCondition", "Kit", "KitDate",
            "InHospitalFreezeTimeInterval"
        ]
        df_info[report_cols].to_csv(f"{self.d_info}/{self.prefix}.all.info.list", sep="\t", index=False)
        df_info.to_csv(f"{self.d_info}/{self.prefix}.all.full.info.list", sep="\t", index=False)
        df_info[["SampleID"]].to_csv(f"{self.d_info}/{self.prefix}.all.id.list", sep="\t", index=False, header=None)
        for ds_name, df_g in df_info[report_cols].groupby("Dataset"):
            df_g.to_csv(f"{self.d_info}/{self.prefix}.{ds_name}.info.list", sep="\t", index=False)
            # 平台配对检查，防止漏检
            if self.paired_mode:
                df_warn = self.fetch_paired(df_g)
                df_warn.to_csv(f"{self.d_output}/{self.prefix}.paired.warn.tsv", sep="\t", index=False)

        # 得到可能被遗漏的样本信息
        df_leave_out = df_all[~(df_all.SampleID.isin(df_exclude.SampleID) | df_all.SampleID.isin(df_info.SampleID))]
        df_leave_out.to_csv(f"{self.d_output}/{self.prefix}.select.LeaverOut.list", sep="\t", index=False)

        # 统计本次数据集，各个项目的样本信息
        df_stat = df_info.groupby(["Dataset", "SelectGroup", "ProjectID", "GroupLevel1", "GroupLevel2", "GroupLevel3", "GroupLevel4", "GroupLevel5", "GroupLevel6"]).size().reset_index()
        df_stat = df_stat.rename(columns={0: "Count"})
        df_stat = df_stat.sort_values(by=["Dataset", "GroupLevel1", "SelectGroup"])
        df_stat["descript"] = df_stat.ProjectID.apply(lambda x: self.get_desc.get(x))
        df_stat["SelectNote"] = df_stat.SelectGroup.apply(lambda x: self.conf["select_note"].get(x, ""))
        df_stat.to_csv(f"{self.d_output}/{self.prefix}.stat.tsv", sep="\t", index=False)

        df_stat_simple = df_info.groupby(["Dataset", "SelectGroup", "GroupLevel1", "GroupLevel2"]).size().reset_index()
        df_stat_simple = df_stat_simple.rename(columns={0: "Count"})
        df_stat_simple = df_stat_simple.sort_values(by=["Dataset", "GroupLevel1", "SelectGroup"])
        df_stat_simple = pd.pivot(df_stat_simple, index=["GroupLevel1", "GroupLevel2", "SelectGroup"], columns="Dataset", values="Count").reset_index()
        df_stat_simple["descript"] = df_stat_simple.SelectGroup.apply(lambda x: self.get_desc.get(x))
        df_stat_simple["SelectNote"] = df_stat_simple.SelectGroup.apply(lambda x: self.conf["select_note"].get(x, ""))
        df_stat_simple = df_stat_simple.fillna(0)
        df_stat_simple.to_csv(f"{self.d_output}/{self.prefix}.stat.simple.tsv", sep="\t", index=False)

        self.client.close()

    def fetch_paired(self, df_info):
        # 以样本第一字段分组，筛选出具有两个以上的SelectGroup信息，并且不同SelectGroup信息对应的Platform也不一样的样本。
        meta_list = df_info.copy()
        meta_list["shortID"] = meta_list["SampleID"].apply(lambda x: x.split("-")[0])
        # 计算每个样本对应平台和分组数量，根据必要条件“分组和平台均大于2种”进行初步筛选
        filtered_samples = meta_list.groupby('shortID').apply(lambda group: group['SelectGroup'].nunique() >= 2 and group['Platforms'].nunique() >= 2)
        candidate_samples = meta_list.set_index("shortID")[["SampleID", "Platforms", "ProjectID", "SelectGroup", "SampleGroup"]].loc[filtered_samples]
        return candidate_samples.sort_index()

    def get_paired_mode(self, df_t, mode):
        """ 从数据集中，筛选出指定配对关系的样本

        :param df_t:
        :param mode: 配对模式。
                     [single]: 单样本模式，
                     [paired]: 意义配对样本模式
                     ["MGI 2000", "MGI T7"]: 指定平台的配对模式
        :return: dataframe
        """

        df_rslt = df_t.copy()  # 最后返回的结果文件
        df_rslt["shortID"] = df_rslt.SampleID.apply(lambda x: x.split("-")[0])
        df_rslt.to_csv(f"/dssg/home/sheny/test/124.tsv", sep="\t", index=False)

        # 确定是否配对

        df_g = df_rslt.groupby("shortID")["Platforms"].unique().reset_index()
        df_g["mode"] = df_g.Platforms.apply(lambda x: "single" if len(x) <= 1 else "paired")
        if df_rslt.iloc[0]["ProjectID"] == "KZ28" and mode == ["single"]:
            df_g.to_csv(f"/dssg/home/sheny/test/125.tsv", sep="\t", index=False)

        if mode == ["single"]:
            sample_list = df_g[df_g["mode"] == "single"]["shortID"].tolist()
        elif mode == ["paired"]:
            sample_list = df_g[df_g["mode"] == "paired"]["shortID"].tolist()
        else:
            mode = "|".join(sorted(mode))
            df_g["Platforms"] = df_g["Platforms"].apply(lambda x: "|".join(sorted(x)))
            sample_list = df_g[df_g.Platforms == mode]["shortID"].tolist()

        df_rslt = df_rslt[df_rslt.shortID.isin(sample_list)]
        df_rslt.pop("shortID")

        return df_rslt

    @property
    def get_desc(self):
        """数据库中，关于各个项目的说明"""

        data = {d["ProjectID"]: d["Institution"] for d in self.col_desc.find({})}
        return data

    @staticmethod
    def get_response(s):
        """根据group level确定样本分类"""

        if s.GroupLevel1 == "Cancer":
            return "Cancer"
        elif s.GroupLevel1 == "Healthy":
            return "Healthy"
        elif s.GroupLevel1 == "Disease":
            if s.GroupLevel3 == "CRA":
                return "Cancer"
            else:
                return "Healthy"
        else:
            raise ValueError(f"can not confirm response: {s}")

    @staticmethod
    def outdir(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p

    def __del__(self):
        self.client.close()


if __name__ == '__main__':
    PartitionDataset(
        # f_conf="/dssg/home/sheny/MyProject/gsml/config/partition_dataset.yaml",
        f_conf="/dssg/home/sheny/MyProject/gsml/config/partition_dataset/PanCancer/pd_panCancer_2022-10-30.yaml",
        d_output="/dssg/home/sheny/test/split_dataset"
    ).partitioning()
