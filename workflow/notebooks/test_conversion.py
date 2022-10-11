# %%

# %%
# 1. set file to load
# 2. load it using neurochat, and using NWB, and using SIMURAN
# 3. assert same for a firing map, and an LFP spectrum
# 4. check NWB spikes against TINT.

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr

# %%
filepath = r"E:\Repos\atn-sub-lfp-workflow\workflow\notebooks\test\nwbfiles\LSubRet1--recording--+maze--04092017_first trial--S3--04092017_LSubRet1_+maze_trial_1_3.nwb"

loader = smr.loader_from_string("nwb")
recording = smr.Recording(source_file=filepath, loader=loader)
recording.load()

# %%
nwbfile = recording.data

spikes = nwbfile.processing["spikes"]["times"].to_dataframe()
waveforms = nwbfile.processing["spikes"]["waveforms"].to_dataframe()

avg_waveforms = []

for index, row in waveforms.iterrows():
    if row[0].startswith("2"):
        num_spikes = row[1]
        waveform_data = row[2][: num_spikes * 50]
        data = waveform_data.reshape(num_spikes, 50)
        avg_waveforms.append(np.mean(data, axis=0))


# %%
fig, axes = plt.subplots(4, 1)
for ax, wave in zip(axes, avg_waveforms):
    ax.plot(wave)
plt.show()
