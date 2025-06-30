#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/9/27 15:39
# @File     : cmd_tabulate.py
# @Project  : gsml
#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author   : shenny
# @Time     : 2022/7/1 17:54
# @File     : cmd_split_dataset.py
# @Project  : gsml

"""生成频数表"""

import click

from module.tabulate import tabulate


__all__ = ["cli_tabulate"]


@click.group()
def cli_tabulate():
    pass


@cli_tabulate.command("tabulate")
@click.option("-i", "--file", required=True, help="Path to info file")
@click.option("-s", "--stat_cols", required=True, help="Indicates the field whose statistics are to be collected")
@click.option("-t", "--stat_types", required=True, help="Type of stats fields. must be str or float")
@click.option("-h", "--header_col",required=True, help="The field corresponding to the table header")
@click.option("-d", "--digits", type=click.INT, default=1, show_default=True, help="The decimal point preserves the number of digits")
@click.option("-o", "--f_output", help="Path of result file")
def cmd_tabulate(**kwargs):
    """Frequency table statistics"""

    kwargs["stat_cols"] = kwargs["stat_cols"].split(",")
    kwargs["stat_types"] = kwargs["stat_types"].split(",")

    if len(kwargs["stat_types"]) != len(kwargs["stat_cols"]):
        raise  ValueError(f"Must same length between stats_cols and stats_types")

    if {"str", "float"} | set(kwargs["stat_types"]) != {"str", "float"}:
        raise ValueError("stat_types must be str of float")

    tabulate(**kwargs)
