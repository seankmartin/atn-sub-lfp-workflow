# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

# %%
loader = smr.loader("nwb")
path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed\CSubRet4--recording--small sq up down--01122017--S1_small sq up--01122017_CSR4_smallsqupdown_up_1_1.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()
nwbfile = recording.data
config = smr.config_from_file(parent.parent / "config" / "simuran_params.yml")

# %% Grab normalised_lfp and non-normalised_lfp
lfp_data = nwbfile.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
non_norm_lfp_data = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].data[:].T
eeg_rate = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
egf_lfp_data = (
    nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"].data[:].T
)
egf_rate = nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"].rate
average_signal = nwbfile.processing["average_lfp"]


# %%
from frequency_analysis import calculate_psd


def filter_non_norm(sig, rate):
    return (
        smr.Eeg.from_numpy(sig, rate)
        .filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
        .samples,
        rate,
    )


sigs = [(non_norm_lfp_data[0], eeg_rate), (egf_lfp_data[0], egf_rate)]
sigs = [filter_non_norm(sig[0], sig[1]) for sig in sigs]


for i, sig in enumerate(sigs):
    sigs[i] = filter_non_norm(sig[0], sig[1])

sigs.append((lfp_data[0], eeg_rate))

pxx_list = [
    calculate_psd(sig[0], sig[1], config["fmin"], config["fmax"], scale="decibels") for sig in sigs
]

fig, ax = plt.subplots(3, 1)

cols = "rgb"
for (pxx_vals, a, c) in zip(pxx_list, ax, cols):
    a.plot(pxx_vals[0], pxx_vals[1], c=c)

plt.show()

# %%
