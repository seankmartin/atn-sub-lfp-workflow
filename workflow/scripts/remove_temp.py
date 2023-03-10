from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file


def main(path_to_all, out_name):
    df_all = df_from_file(path_to_all)
    df_all.drop(
        ["directory", "filename", "mapping", "light", "optional_grouping"],
        axis=1,
        inplace=True,
    )
    df_to_file(df_all, out_name)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        Path(snakemake.input[0]),
        Path(snakemake.output[0]),
    )
