#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/1 11:20

import json
import os
import random
import re
from subprocess import check_output, Popen
import time

import h2o
import pandas as pd
import yaml

from module.error import *

__all__ = ["connect", "init", "cluster", "close"]


def connect(url=None, ip=None, port=None, **kwargs):
    """连接到一个h2o集群"""

    h2o.connect(url=url, ip=ip, port=port, **kwargs)


def close():
    """关闭一个h2o server"""

    import h2o

    h2o.cluster().shutdown()


def init(retry=5, nthreads=1, max_mem_size="4000M",  **kwargs):
    """ 尝试连接到本地服务器，如果不成功，则启动新服务器并连接。初始化一个本地h2o服务

    :param max_mem_size: 最大内存
    :param nthreads: 最大线程数
    :param retry: 重试的次数
    :param kwargs: 兼容h2o.init的所有参数
    :return:
    """

    times = 0
    while times <= retry:
        time.sleep(10)
        try:
            kwargs["port"] = random.randint(40000, 65535)
            h2o.init(nthreads=nthreads, max_mem_size=max_mem_size, **kwargs)
        except:
            pass
        else:
            break


def cluster(name, workdir, n_nodes, threads=5, memory="14g", version=None, port=None):
    """
    :param name:
    :param workdir:
    :param n_nodes:
    :param threads:
    :param memory:
    :param version:
    :param port:
    :return:
    """

    port = port or random.randint(40000, 65535)
    ins = Cluster(name=name, workdir=workdir)
    ins.start(n_nodes=n_nodes, threads=threads, memory=memory, version=version, port=port)


class Cluster(object):
    """ h2o集群的部署与相关操作

    :parameter
        name: 集群的名称
        workdir: 集群工作的目录（若想创建新的h2o集群，则此目录必须为空）

    """

    def __init__(self, name, workdir):
        self.name = name
        self.workdir = self._outdir(f"{workdir}/{name}")
        self.d_log = self._outdir(f"{self.workdir}/log")
        self.f_job = f"{self.workdir}/{name}.cluster.info.txt"

        self.d_gsml = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.d_bin = f"{self.d_gsml}/bin"
        self.d_conf = f"{self.d_gsml}/config"

        if os.path.exists(self.f_job):
            raise DirNotEmpty(f"workdir is not empty")

    def start(self, n_nodes, threads, memory, version=None, port=None):
        """ 启动h2o集群

        :param port:
        :param n_nodes:
        :param threads:
        :param memory:
        :param version:
        :return:
        """

        # 确定集群h2o版本
        if not version:
            h2o_java = f"{self.d_bin}/h2o/h2o/h2o.jar"
        else:
            h2o_java = f"{self.d_bin}/h2o/h2o-{version}/h2o.jar"

        # 启动任务
        logger.info(f"start h2o server: nodes {n_nodes}; threads {threads}; memory {memory};")
        conf = yaml.load(open(f"{self.d_conf}/cluster_ip.yaml"), Loader=yaml.FullLoader)
        run_nodes = []
        times = 0

        while True:
            for node in conf["base"]:
                cmd = f"java -Xmx{memory} -jar {h2o_java} " \
                      f"-name {self.name} " \
                      f"-ip {node['ip']} " \
                      f"-port {port} " \
                      f"-nthreads {threads} " \
                      f"-log_dir {self.d_log} "

                f_err = f"{self.d_log}/{node['name']}.err"
                f_out = f"{self.d_log}/{node['name']}.out"
                job_id = self.bsub(cmd, cpu=threads, job_name=f"H2oServer_{self.name}", nodes=node["name"],
                                   err_output=f_err, screen_output=f_out)
                if job_id:
                    logger.info(f"run on nodes: <{node['name']}> <{job_id}>")
                    run_nodes.append({"node": node['name'], "ip": node['ip'], "port": port, "jobid": job_id})

                times += 1
                if len(run_nodes) >= n_nodes or times >= 500:
                    break
            if len(run_nodes) >= n_nodes or times >= 500:
                break

        df_run = pd.DataFrame(run_nodes)
        df_run.to_csv(self.f_job, sep="\t")
        logger.info(f"h2o server(H2oServer_{self.name}) is run: {df_run.iloc[0]['ip']}:{df_run.iloc[0]['port']}")

    @staticmethod
    def bsub(command, cpu=5, job_name=None, queue="normal", nodes=None, err_output=None, screen_output=None,
             max_wait_sec=30):
        """ 将任务提交至集群，并确定任务已经run之后再返回任务状态

        :param nodes:
        :param command:
        :param cpu:
        :param job_name:
        :param queue:
        :param err_output:
        :param screen_output:
        :param max_wait_sec:
        :return:
        """

        cmd = f'bsub -J {job_name} -q {queue} -n {cpu} -R "span[hosts=1]" '
        if nodes:
            cmd += f"-m {nodes} "
        if err_output:
            cmd += f"-eo {err_output} "
        if screen_output:
            cmd += f"-oo {screen_output} "
        cmd += f'"{command}"'

        # 提交任务并返回任务状态
        _report = check_output(cmd, shell=True, encoding="utf-8")
        job_id = re.findall(f"<(\d+)>", _report)[0]

        start = time.time()
        status = ""
        while status not in ["RUN", "EXIT"] or time.time() - start < max_wait_sec:
            time.sleep(1)
            _stat = check_output(f'export LSB_BJOBS_FORMAT="STAT" && bjobs {job_id} -json', shell=True, encoding="utf-8")
            status = json.loads(_stat)["RECORDS"][0]["STAT"]

        if status != "RUN":
            Popen(f"bkill {job_id}", shell=True)
            return None
        else:
            return job_id

    def close(self):
        """关闭"""

        df_job = pd.read_csv(self.f_job, sep="\t")
        for job_id in df_job.jobid:
            Popen(f"bkill {job_id}", shell=True)
        logger.info(f"h2o server(H2oServer_{self.name}) is closed")

    @staticmethod
    def _outdir(p):
        if not os.path.exists(p):
            try:
                os.makedirs(p)
            except:
                pass
        return p
