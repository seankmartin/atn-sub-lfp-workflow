# %%
%load_ext autoreload
%autoreload 2
import logging
import sys
from pathlib import Path

import pynwb
import simuran as smr
from skm_pyutils.table import df_from_file, filter_table

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

from convert_to_nwb import main

module_logger = logging.getLogger("simuran.custom.convert_to_nwb")
module_logger.setLevel(logging.DEBUG)
module_logger.addHandler(logging.StreamHandler(sys.stdout))

# %%
table_path = r"E:\Temp\atn-sub-lfp-workflow\results\subret_recordings.csv"
config_path = r"E:\Temp\atn-sub-lfp-workflow\config\simuran_params.yml"
filename = "04122017_CSR4_sleep_1_1_awake.set"
# filename = "04092017_CSubRet1_+maze_trial_1_1.set"
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
files = main(table, config, filter_, output_dir, out_name, overwrite=True, debug=False)

# %%
nwbfile = files[0]
nwb_io = pynwb.NWBHDF5IO(nwbfile, "r")
nwbfile = nwb_io.read()
print(nwbfile.electrodes.to_dataframe())
print(nwbfile.electrode_groups.keys())

# %%
