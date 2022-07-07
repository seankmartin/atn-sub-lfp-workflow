import os
from pathlib import Path

import simuran as smr
from pandas import DataFrame
from skm_pyutils.table import df_from_file, filter_table, list_to_df

from ..convert_to_nwb import main


def convert_tmaze_data(
    main_table_path,
    t_maze_filter_path,
    t_maze_times_path,
    main_cfg_path,
    output_dir,
    overwrite=False,
):
    df = df_from_file(main_table_path)
    filter_cfg = smr.config_from_file(t_maze_filter_path)
    t_maze_times = df_from_file(t_maze_times_path)
    out_name = f"{Path(t_maze_times_path).stem}_nwb.csv"
    cfg = smr.config_from_file(main_cfg_path)

    filtered_df = filter_table(df, filter_cfg)
    new_tmaze_df = modify_tmaze_times(t_maze_times, cfg["cfg_base_dir"])
    merged_df = merge_times_files(filtered_df, new_tmaze_df)
    from skm_pyutils.table import df_to_file

    df_to_file(merged_df, "testtmaze.csv")
    exit(-1)

    main(
        merged_df,
        cfg,
        filter_=None,
        output_dir=output_dir,
        out_name=out_name,
        overwrite=overwrite,
    )


def change_mapping(s):
    bits = os.path.splitext(s)
    return bits[0] + "-no-cells" + bits[1]


def merge_times_files(files_df: DataFrame, times_df: DataFrame):
    files_df.merge(
        times_df,
        how="inner",
        on=["directory", "filename"],
        validate="one_to_one",
        suffixes=(None, "_x"),
    )
    files_df.drop(
        ["directory_x", "filename_x", "mapping_x", "animal"], axis=1, inplace=True
    )
    files_df.loc[:, "mapping"] = files_df["mapping"].apply(change_mapping)
    return files_df


def modify_tmaze_times(df, base_dir):
    new_df_list = []
    headers = tmaze_headers()

    for row in df.itertuples():
        if row.test == "first":
            splits = row.location.split("--")
            directory = os.path.join(base_dir, os.sep.join(splits[:-1]))
            filename = splits[-1]
            ilist = [
                directory,
                filename,
                row.session,
                row.trial,
                row.passed,
            ]
        part = [
            row.start,
            row.choice,
            row.end,
        ]
        ilist.extend(part)

        if row.test == "second":
            new_df_list.append(ilist)

    return list_to_df(new_df_list, headers)


def tmaze_headers():
    return [
        "directory",
        "filename",
        "session",
        "trial",
        "passed",
        "start1",
        "choice1",
        "end1",
        "start2",
        "passed2",
        "choice2",
        "end2",
        "mapping",
    ]


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    convert_tmaze_data(
        snakemake.input[0],
        snakemake.input[1],
        snakemake.config["tmaze_filter"],
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]).parent,
        snakemake.config["overwrite_nwb"],
    )
