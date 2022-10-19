# %%
import sys
from pathlib import Path

import numpy as np
import simuran as smr
from skm_pyutils.table import df_from_file

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

from analyse_sleep import spindle_control

# %%
fname = parent.parent / "results" / "other_process.csv"
df = df_from_file(fname)
df = df[
    df["nwb_file"]
    == r"results\processed\CSubRet4--recording--sleep--CSR4_sleep_07122017--S3--07122017_CSR4_sleep_2_sleep.nwb"
]
df["nwb_file"] = [
    parent.parent
    / "results\processed\CSubRet4--recording--sleep--CSR4_sleep_07122017--S3--07122017_CSR4_sleep_2_sleep.nwb"
]
loader = smr.loader_from_string("nwb")
rc = smr.RecordingContainer.from_table(df, loader=loader)
config = smr.config_from_file(parent.parent / "config" / "simuran_params.yml")

# %%
r = rc.load(0)
spindles, resting_array = spindle_control(r, config)

# %%
print(spindles["SUB"])
spindles_sub = spindles["SUB"]
spindles_chan1 = spindles_sub[spindles_sub["Channel"] == "SUB_0"]
spindles_chan2 = spindles_sub[spindles_sub["Channel"] == "SUB_1"]
print(spindles_chan1, spindles_chan2)

# %%
spindles, resting_array2 = spindle_control(r, config, True)
print(spindles["SUB"])

# %%
assert np.all(np.isclose(resting_array, resting_array2))

# %%
