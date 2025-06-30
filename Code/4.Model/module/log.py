#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 21:57
# @Author  : shenny
# @File    : log.py
# @Software: PyCharm

"""日志模块"""

import logging
import coloredlogs

coloredlogs.install(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

__all__ = ["debug", "info", "warning", "error"]

def debug(message, verbose):
    if verbose == 0:
        logging.debug(message)

def info(message, verbose):
    if verbose <= 1:
        logging.info(message)

def warning(message, verbose):
    if verbose <= 2:
        logging.warning(message)

def error(message, verbose):
    if verbose <= 3:
        logging.error(message)
