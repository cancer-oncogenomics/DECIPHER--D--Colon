#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/2 11:59
# @File     : submit_lsf.py
# @Project  : gsml

import os
import re
import subprocess
import time
import logging
import uuid

import coloredlogs

from module.error import *

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def submit_lsf(commands: list, d_output, nthreads=10, wait=60, force=False):
    """ 将命令提交至lsf运行

    :param wait:
    :param commands: [("", ""), ("", "")]
    :param d_output:
    :param nthreads:
    :return:
    """

    # 判断目录是否具有可写权限
    if not os.access(os.path.dirname(d_output), os.W_OK):
        d_output = f"~/.log/gsml/{uuid.uuid1()}"

    # 生成命令脚本
    if not os.path.exists(d_output):
        try:
            os.makedirs(d_output)
        except Exception as error:
            print(error)
            pass

    # 提交命令
    check_list = []
    for name, cmd in commands:
        logger.info(f"submit job {name}")
        f_sh = f"{d_output}/{name}.sh"
        f_log = f"{d_output}/{name}.log"
        f_error = f"{d_output}/{name}.error"
        f_done = f"{d_output}/{name}.done"
        f_failed = f"{d_output}/{name}.failed"

        if os.path.exists(f_done) and not force:
            print(f_done)
            continue
        else:
            with open(f_sh, "w") as fw:
                fw.write(cmd)
            run_cmd = f"bash {f_sh} 1> {f_log} 2> {f_error} && touch {f_done} || touch {f_failed}"
            run_cmd = f'bsub -J {name} -n {nthreads} -R "span[hosts=1]" "{run_cmd}"'
            response = subprocess.check_output(run_cmd, shell=True, encoding="utf-8")
            job_id = re.findall('<(\d+)>', response)[0]
            check_list.append({"JobID": job_id, "f_done": f_done, "f_failed": f_failed, "job_name": name})

    # 等待任务完成
    failed = []
    while True:
        logger.info(f"wait job")
        if not check_list:
            break

        job = check_list[0]
        response = subprocess.check_output(f"bjobs {job['JobID']}", shell=True, encoding="utf-8")

        if "PEND" not in response and "RUN" not in response:
            if not os.path.exists(job["f_done"]):
                msg = f"job run error: ({job['job_name']})"
                logger.error(msg)
                failed.append(job["job_name"])
            else:
                logger.info(f"job done: {job['job_name']}")
            check_list.pop(0)
        time.sleep(wait)

    if failed:
        msg = f"some job run error: {','.join(failed)}"
        logger.error(msg)
        raise LsfJobError(msg)
    else:
        logger.info(f"all jobs done")
