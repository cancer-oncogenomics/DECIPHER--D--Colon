"""Fit Mutational Signatures (MS) feature"""

import pandas as pd
import click


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--context", help="MC feature file for signature fitting")
@click.option('--f_output', help="Output")
def prep_fit_signatures(context, f_output):
    data = pd.read_csv(context)
    # transpose to long format and rename columns using first row
    data = data.T.reset_index()
    data.columns = ["context"] + data.iloc[0, 1:].to_list()
    data = data.iloc[1:, :]
    # transform MC context colnames to the format of 3-nt or mut
    # 3-nt: A[C>A]A
    # mut: [C>A]AA
    data["context_3nt"] = data["context"].apply(lambda x: x[-4] + '[' + x[-3] + '>' + x[-2] + ']' + x[-1])
    data["context_mut"] = data["context"].apply(lambda x: '[' + x[-3] + '>' + x[-2] + ']' + x[-4] + x[-1])
    # sort by SBS mutation for signature fit function in R (MutationalPatterns)
    data = data.sort_values("context_mut")
    data = data.set_index("context_3nt", drop=True)
    data = data.drop(columns=["context_mut"])
    data.to_csv(f_output, sep="\t", index=True, header=True)


if __name__ == "__main__":
    prep_fit_signatures()
