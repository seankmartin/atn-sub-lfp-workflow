# %%
import sys
from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file, filter_table

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

from convert_to_nwb import main

# %%

table_path = r"E:\Repos\atn-sub-lfp-workflow\results\subret_recordings.csv"
config_path = r"E:\Repos\atn-sub-lfp-workflow\config\simuran_params.yml"
filter_path = r"E:\Repos\atn-sub-lfp-workflow\workflow\sheets\muscimol_cells.csv"

# %%
config = smr.ParamHandler(source_file=config_path)
table = df_from_file(table_path)
id_table = df_from_file(filter_path)
filter_ = {"filename": id_table["filename"].values}
filtered_table = filter_table(table, filter_)
filtered_table.merge(
    id_table,
    how="left",
    on="filename",
    validate="one_to_one",
    suffixes=(None, "_x"),
)
if "directory_x" in filtered_table.columns:
    filtered_table.drop("directory_x", inplace=True)

# %%
output_dir = Path(r"E:\Repos\atn-sub-lfp-workflow")
out_name = r"test-can-convert.csv"
main(filtered_table, config, None, output_dir, out_name, overwrite=False)

# %%
