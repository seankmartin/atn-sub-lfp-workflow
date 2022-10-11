# %%
%load_ext autoreload
%autoreload 2
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
filename = "04092017_LSubRet1_+maze_trial_1_3.set"
output_dir = Path("test")
out_name = "test.csv"

# %%
config = smr.ParamHandler(source_file=config_path)
table = df_from_file(table_path)
filter_ = {"filename": [filename]}
filtered_table = filter_table(table, filter_)

# %%
config["loader_kwargs"]["pos_extension"] = ".pos"
loader = smr.loader("neurochat", **config["loader_kwargs"])
rc = smr.RecordingContainer.from_table(filtered_table, loader=loader)
r = rc[0]

# %%
main(table, config, filter_, output_dir, out_name, overwrite=True, debug=False)

# %%
