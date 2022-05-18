from email.mime import base
from pathlib import Path

from skm_pyutils.py_config import read_python
from skm_pyutils.py_path import (get_all_files_in_dir, get_dirs_matching_regex,
                                 remove_empty_dirs_and_caches)
from skm_pyutils.py_table import df_from_file, df_to_file, list_to_df

here = Path(__file__).resolve().parent


def main(dirname, path_to_csv, output_path):
    fnames = get_all_files_in_dir(here / "batch_params", ext=".py")
    df = df_from_file(path_to_csv)

    out_list = []
    for fname in fnames:
        params = read_python(fname, dirname)["params"]
        mapping_file = Path(params["mapping_file"]).name
        matching_dirs = get_dirs_matching_regex(
            params["start_dir"], params["regex_filters"]
        )
        filtered_dirs = remove_empty_dirs_and_caches(matching_dirs)
        dirs_with_set = []
        for dir_ in filtered_dirs:
            all_files = get_all_files_in_dir(
                dir_, ext=".set", case_sensitive_ext=True, return_absolute=True
            )
            if len(all_files) == 1:
                dirs_with_set.append(dir_)

        for dir_ in dirs_with_set:
            out_list.append((dir_, Path(fname).stem, mapping_file))

    df_to_merge = list_to_df(
        out_list, transpose=False, headers=["directory", "recording_type", "mapping"]
    )
    # duplicates = df[df.duplicated("directory", keep=False)]
    merged_df = df.merge(
        df_to_merge,
        how="left",
        on="directory",
        validate="many_to_one",
        suffixes=("_x", None),
    )
    merged_df["mapping"].fillna(merged_df["mapping_x"], inplace=True)
    del merged_df["mapping_x"]

    base_mapping_dir = here / "recording_mappings"
    new_mapping = []
    for idx, row in merged_df.iterrows():
        mapping = row["mapping"]
        mapping_location = base_mapping_dir / mapping
        if mapping_location.is_file():
            new_mapping.append(mapping_location)
        else:
            new_mapping.append(mapping)
    merged_df["mapping"] = new_mapping

    df_to_file(merged_df, output_path)


if __name__ == "__main__":
    main(snakemake.config["data_directory"], snakemake.input[0], snakemake.output[0])
