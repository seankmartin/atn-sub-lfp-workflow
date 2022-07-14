# %%
%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file, filter_table

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

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
config["loader_kwargs"]["pos_extension"] = ".pos"
loader = smr.loader("neurochat", **config["loader_kwargs"])
rc = smr.RecordingContainer.from_table(filtered_table, loader=loader)
r = rc.load(0)
print(r.data)

# %%
r.data["spatial"].direction

# %%
from convert_to_nwb import convert_recording_to_nwb
# %%
from simuran.loaders.nc_loader import NCLoader

# %%

convert_recording_to_nwb(r, config["cfg_base_dir"])

# %%
pth = r"E:\Repos\atn-sub-lfp-workflow\test.csv"
df = df_from_file(pth)
rc = smr.RecordingContainer.from_table(df, loader=loader)

# %%
r = rc.load(0)
print(r.data)
r.data["spatial"].direction

# %%
convert_recording_to_nwb(r, config["cfg_base_dir"])

# %%
