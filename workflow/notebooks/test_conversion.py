# %%

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
filepath = r"D:\atn-sub-lfp-workflow\results\nwbfiles\CSubRet4--recording--big sq_1wall_2wall--27112017--S1_big sq--27112017_CSR4_bigsq_1_1.nwb"

loader = smr.loader_from_string("nwb")
recording = smr.Recording(source_file=filepath, loader=loader)
recording.load()

# %%
def nwb_analysis(nwbfile):
    lfp_data = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].data[:].T
    behavior = nwbfile.processing["behavior"]
    position = behavior["Position"]["SpatialSeries"].data[:].T
    running_speed = behavior["running_speed"].data[:]
    spikes_data = nwbfile.processing["spikes"]["times"].to_dataframe()
    waveforms = nwbfile.processing["spikes"]["waveforms"].to_dataframe()
    rate = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    for index, row in waveforms.iterrows():
        if row[0].startswith("4"):
            num_spikes = row[1]
            waveform_data = row[2][: num_spikes * 50]
            data = waveform_data.reshape(num_spikes, 50)
            break
    for index, row in spikes_data.iterrows():
        if row[0].startswith("4"):
            num_spikes = row[1]
            times_data = row[2][:num_spikes]
            break
    plotting(lfp_data[2], rate, position, running_speed, times_data, data)


def plotting(lfp, rate, position, speed, unit_times, unit_wave):
    plt.plot(welch(lfp, fs=rate, nperseg=2 * rate, scaling="density")[1])
    plt.show()
    plt.close()
    plt.hist2d(position[0], position[1])
    plt.show()
    plt.close()
    plt.hist(speed)
    plt.show()
    plt.close()
    plt.hist(unit_times)
    plt.show()
    plt.close()
    plt.plot(np.mean(unit_wave, axis=0))
    plt.show()
    plt.close()


# %%
def nc_analysis(filepaths):
    lfp_file, pos_file, spike_file = filepaths
    ndata = NData()
    ndata.set_spike_file(spike_file)
    ndata.set_spatial_file(pos_file)
    ndata.set_lfp_file(lfp_file)
    ndata.load()

    lfp_data = ndata.lfp.get_samples()
    lfp_rate = ndata.lfp.get_sampling_rate()
    position_data = np.array([ndata.spatial.get_pos_x(), ndata.spatial.get_pos_y()])
    speed = ndata.spatial.get_speed()
    unit_times = ndata.spike._timestamp
    unit_waves = np.array([v for v in ndata.spike._waveform.values()])[0]
    plotting(lfp_data, lfp_rate, position_data, speed, unit_times, unit_waves)


# %%
nwbfile = recording.data
nwb_analysis(nwbfile)

# %%
dir_ = Path(
    r"H:\SubRet_recordings_imaging\CSubRet4\recording\big sq_1wall_2wall\27112017\S1_big sq"
)
name = "27112017_CSR4_bigsq_1_1"
fpaths = [dir_ / f"{name}.eeg3", dir_ / f"{name}_4.txt", dir_ / f"{name}.4"]
fpaths = [str(f) for f in fpaths]
nc_analysis(fpaths)
plt.show()

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
