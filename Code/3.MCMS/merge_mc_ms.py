#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 15:17
# @Author  : shenny
# @File    : merge_mc_ms.py
# @Software: PyCharm

import click
import pandas as pd


@click.command()
@click.option("--f_mc", required=True)
@click.option("--f_ms", required=True)
@click.option("--f_mc_out")
@click.option("--f_mcms_out", required=True)
def cli(f_mc, f_ms, f_mc_out, f_mcms_out):

    df_mc = pd.read_csv(f_mc)
    df_mc = df_mc.rename(mapper=lambda x: x.replace(".SNP_in", "").replace(".SNP_out", ""), axis=1)
    if f_mc_out:
        df_mc.to_csv(f_mc_out, index=False)

    df_mc = df_mc.rename(mapper=lambda x: x.replace("MC.", "MCMS."), axis=1)
    df_ms = pd.read_csv(f_ms)
    df_ms = df_ms.rename(mapper=lambda x: x.replace(".SNP_in", "").replace(".SNP_out", ""), axis=1)
    df_ms = df_ms.rename(mapper=lambda x: x.replace("MS.", "MCMS."), axis=1)
    df_mcms = pd.merge(df_mc, df_ms, on="SampleID", how="inner")
    df_mcms.to_csv(f_mcms_out, index=False)


if __name__ == '__main__':
    cli()
