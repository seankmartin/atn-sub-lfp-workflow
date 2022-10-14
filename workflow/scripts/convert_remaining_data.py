from pathlib import Path

import pandas as pd
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file

from convert_to_nwb import main as convert_table_to_nwb
from process_lfp import main as process_tables


def main(path_to_all, path_to_converted, config_path, outputs, num_cpus=1):
    df_all = df_from_file(path_to_all)
    df_converted = df_from_file(path_to_converted)
    config = smr.config_from_file(config_path)

    df_merged = df_all.merge(
        df_converted.drop_duplicates(),
        on=["directory", "filename"],
        how="left",
        indicator=True,
    )
    to_convert = df_all[df_merged["_merge"] == "left_only"]

    # TEMP for now, let us try to get sleep working
    to_convert = to_convert[to_convert["sleep"] == 1]

    # Ignore those with no mapping and print warning
    no_mapping = to_convert[to_convert["mapping"] == "no_mapping"]
    if len(no_mapping) > 0:
        print(f"WARNING ignoring {len(no_mapping)} files with no mapping")
        rats = set(no_mapping["rat"])
        print(f"This includes rats {rats}")
    to_convert = to_convert[to_convert["mapping"] != "no_mapping"]

    out_name1, out_name2, out_name3 = outputs
    output_dir = out_name1.parent
    out_name = out_name1.name
    convert_table_to_nwb(to_convert, config, None, output_dir, out_name)

    process_tables(
        [out_name1], config_path, ["temp.csv", out_name2], num_cpus, overwrite=False
    )

    dfs = [df_converted, df_from_file(out_name2)]
    final_df = pd.concat(dfs, ignore_index=True)
    df_to_file(final_df, out_name3)


if __name__ == "__main__":
    try:
        smr.set_only_log_to_file(snakemake.log[0])
        main(
            Path(snakemake.input[0]),
            Path(snakemake.input[1]),
            snakemake.config["simuran_config"],
            [Path(o) for o in snakemake.output],
            snakemake.threads,
        )
    except Exception:
        here = Path(__file__).parent.parent.parent
        input_path1 = here / "results" / "subret_recordings.csv"
        input_path2 = here / "results" / "openfield_processed.csv"
        config_path = here / "config" / "simuran_params.yml"
        fname1 = here / "results" / "other_converted.csv"
        fname2 = here / "results" / "other_processed.csv"
        fname3 = here / "results" / "every_processed_nwb.csv"
        fnames = [fname1, fname2, fname3]
        main(input_path1, input_path2, config_path, fnames)
