# %%
# 1. set file to load
# 2. load it using neurochat, and using NWB, and using SIMURAN
# 3. assert same for a firing map, and an LFP spectrum
# 4. check NWB spikes against TINT.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr
from neurochat.nc_data import NData
from scipy.signal import welch

# %%
filepath = r"D:\atn-sub-lfp-workflow\results\processed\CSubRet4--recording--big sq_1wall_2wall--27112017--S1_big sq--27112017_CSR4_bigsq_1_1.nwb"

loader = smr.loader_from_string("nwb")
recording = smr.Recording(source_file=filepath, loader=loader)
recording.load()

# %%
data = recording.data.processing["average_lfp"]["SUB_avg"].data[:1000]

plt.plot(1000 * data)
plt.show()
