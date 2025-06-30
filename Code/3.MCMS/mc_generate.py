"""Subtract baseline from mutational contexts (MC) summary and generate MC feature file"""

import re
import pandas as pd
import pyreadr
import click


TRI_CONTEXTS = {
    'A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T', 'A[C>G]A', 'A[C>G]C',
    'A[C>G]G', 'A[C>G]T', 'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T',
    'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'A[T>C]A', 'A[T>C]C',
    'A[T>C]G', 'A[T>C]T', 'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T',
    'C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T', 'C[C>G]A', 'C[C>G]C',
    'C[C>G]G', 'C[C>G]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T',
    'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T', 'C[T>C]A', 'C[T>C]C',
    'C[T>C]G', 'C[T>C]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T',
    'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'G[C>G]A', 'G[C>G]C',
    'G[C>G]G', 'G[C>G]T', 'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T',
    'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'G[T>C]A', 'G[T>C]C',
    'G[T>C]G', 'G[T>C]T', 'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T',
    'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T', 'T[C>G]A', 'T[C>G]C',
    'T[C>G]G', 'T[C>G]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T',
    'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 'T[T>C]A', 'T[T>C]C',
    'T[T>C]G', 'T[T>C]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T',
}


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option('--sample', required=True, help="Sample ID")
@click.option('--mutation', required=True, help="GC normalized mutation counts (.rds)")
@click.option('--baseline', required=True, default=None, help="Baseline with depth normalization")
@click.option('--colprefix', required=True, help="Prefix of output column names")
@click.option('--summary', required=True)
@click.option('--level', required=True)
@click.option('--output', required=True)
def subtract_background(sample: str, mutation: str, baseline: str, colprefix: str,
                        summary: str, level: str, output: str):
    """subtract background with/without depth normalization"""

    df_summary = pd.read_csv(summary, sep="\t")
    depth = float(df_summary["MEAN_DEPTH"].iloc[0]) if level == "Raw" else float(re.findall(r"(\d+)X", level)[0])
    dup = round(float(df_summary["DUPLICATE(%)"].iloc[0]), 2)
    dedupdepth = depth * (100 - dup) / 100

    # read GC-normalized mutation counts
    df_n_mut = pyreadr.read_r(mutation)[None]

    # filter out invalid records
    df_n_mut = df_n_mut.loc[~(df_n_mut["context"].str.contains(',')), :]
    df_n_mut = df_n_mut.loc[~(df_n_mut["mut_reads"].isna()), :]
    df_n_mut = df_n_mut.loc[df_n_mut["DP4"] == "0,0,1,1", :]
    # fill in zeros for missing contexts
    if len(df_n_mut) != 96:
        print(f"{sample} missing {96 - len(df_n_mut)} contexts")
        missing_contexts = TRI_CONTEXTS.difference(set(df_n_mut["context"]))
        for ctx in missing_contexts:
            df_n_mut.loc[len(df_n_mut)] = [sample, ctx, 2.0, "0,0,1,1", 0.0]
    # only keep the mutation counts
    df = df_n_mut.loc[:, ["context", "n_mut.normalized"]].set_index("context")
    # depth normalization
    df_base = pd.read_csv(baseline, header=None, index_col=0, names=["base_val"])
    df_sub = pd.merge(df / dedupdepth, df_base, left_index=True, right_index=True, validate="1:1")
    df_sub["corrected_val"] = df_sub.apply(
        lambda row: max(row["n_mut.normalized"] - row["base_val"], 0.0), axis=1
    )
    print(df_sub)
    df_output = df_sub["corrected_val"].to_frame().T.reset_index(drop=True)
    df_output = df_output.rename(
        mapper=lambda x: f"{colprefix}." + re.sub(r'[\[>\]]', '', x), axis=1)
    df_output.insert(0, "SampleID", sample)
    df_output["SampleID"] = sample
    df_output.to_csv(output, index=False)
    print(df_output.iloc[0].tolist())


if __name__ == "__main__":
    subtract_background()
