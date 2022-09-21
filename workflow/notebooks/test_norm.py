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
average_signal = nwbfile.processing["average_lfp"]

# %% This confirms that I am correctly doing a linear transformation
def filter_non_norm(sig):
    return (
        smr.Eeg.from_numpy(sig, 250)
        .filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
        .samples
    )


fig, axes = plt.subplots(2, 1)
sig1 = lfp_data[18]
sig2 = non_norm_lfp_data[18]
axes[0].plot(sig1[:1000])
axes[1].plot(sig2[:1000])
fig.savefig("test2.png", dpi=150)
plt.close(fig)

sig2 = filter_non_norm(sig2)

# %% Check impact of norm on power (there should be none)
from frequency_analysis import calculate_psd

f_norm, Pxx_norm, _ = calculate_psd(sig1, scale="decibels")
f_nnorm, Pxx_nnorm, _ = calculate_psd(sig2, scale="decibels")

assert np.all(np.isclose(f_norm, f_nnorm))
assert np.all(np.isclose(Pxx_norm, Pxx_nnorm))

fig, axes = plt.subplots(2, 1)
axes[0].plot(f_norm, Pxx_norm)
axes[1].plot(f_nnorm, Pxx_nnorm)
fig.savefig("test_psd.png", dpi=200)
plt.close(fig)

# %%
rsc_norm = lfp_data[2]
rsc_non_norm = filter_non_norm(non_norm_lfp_data[2])

fs = 250
import scipy

f, Cxy = scipy.signal.coherence(rsc_norm, sig1, fs, nperseg=2 * fs)
fn, Cxyn = scipy.signal.coherence(rsc_non_norm, sig2, fs, nperseg=2 * fs)

assert np.all(np.isclose(Cxy, Cxyn))

# %%
from fooof import FOOOF

f, V, _ = calculate_psd(sig1, scale="volts")
f1, V1, _ = calculate_psd(sig2, scale="volts")
fo = FOOOF(
    peak_width_limits=[1.0, 8.0],
    max_n_peaks=4,
    min_peak_height=0.2,
    peak_threshold=2.0,
    aperiodic_mode="fixed",
)
res1 = fo.fit(f, V)
fo.report()
res2 = fo.fit(f1, V1)
fo.report()

# %%
from simuran.bridges.neurochat_bridge import signal_to_neurochat

unit_table = recording.data.units.to_dataframe()
spike_train = unit_table.loc[unit_table["tname"] == "TT3_U1"].spike_times.iloc[0]


def compute_spike_lfp(lfp, spike_train, nrep=100):
    np.random.seed(42)
    g_data = lfp.plv(spike_train, mode="bs", fwin=[0, 100], nrep=nrep)
    sta = g_data["STAm"]
    sfc = g_data["SFCm"]
    t = g_data["t"]
    f = g_data["f"]

    return sta, sfc, t, f


nc_sig1 = signal_to_neurochat(smr.Eeg.from_numpy(sig1, fs))
sta, sfc, _, _ = compute_spike_lfp(nc_sig1, spike_train)
sta1, sfc1, _, f = compute_spike_lfp(
    signal_to_neurochat(smr.Eeg.from_numpy(sig2, fs)), spike_train
)

fig, axes = plt.subplots(2, 1)
axes[0].plot(sta1, c="b")
axes[1].plot(sta, c="k")
fig.savefig("test_slfp.png", dpi=200)
plt.close(fig)

assert np.all(np.isclose(sfc1, sfc))

plt.plot(f, sfc)
plt.show()
# %%
