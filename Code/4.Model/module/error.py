#!/usr/bin/env python
# coding: utf-8
# Author：Shen Yi
# Date ：2022/5/7 13:25

import logging

import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class DirNotEmpty(Exception):
    """目录不存在"""

    def __init__(self, message):
        super().__init__(message)
        logger.error(message)


class SampleNotFound(Exception):
    """info和特征合并时，数据不匹配"""

    def __init__(self, message):
        super().__init__(message)
        logger.error(message)


class ArgsError(Exception):
    """参数设置错误"""

    def __init__(self, message):
        super().__init__(message)
        logger.error(message)


class LsfJobError(Exception):
    """lsf提交的任务报错"""

    def __init__(self, message):
        super().__init__(message)
        logger.error(message)

class ColumnsInconsistent(Exception):
    """两个特征的列不一致"""

    def __init__(self, message):
        super().__init__(message)
        logger.error(message)