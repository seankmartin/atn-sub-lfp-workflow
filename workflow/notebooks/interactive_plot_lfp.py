# %%
import itertools
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from skm_pyutils.plot import UnicodeGrabber
from skm_pyutils.table import list_to_df

# %%
loader = smr.loader("nwb")
# path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed\CSR6--screening_small sq--20032018--20032018_CSR6_screening_small sq_8.nwb"
# path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed\LSubRet5--recording--Small sq up_small sq down--01122017--S2_small sq down--01122017_smallsqdownup_down_1_2.nwb"
path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\nwbfiles\CSubRet4--recording--+maze--01122017_t3--S6--01122017_CSubRet4_+maze_t3_6.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()
nwbfile = recording.data

# %% units

# units = nwbfile.units.to_dataframe()
# print(units)
# spike_train = units.

# %%
lfp_data = nwbfile.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
print(lfp_data.shape)
print(lfp_data[16])
average_signal = nwbfile.processing["average_lfp"]
print(average_signal["RSC_avg"].data[:].shape)
print(average_signal)

for k in average_signal.data_interfaces:
    print(k)

plt.plot(lfp_data[18][:1000])
plt.savefig("test2.png", dpi=150)
plt.close()

# %%
def plot_all_signals(recording, output_path, show_bad=True, use_norm=True):
    type_ = "normalised_lfp" if use_norm else "ecephys"
    eeg_array = smr.EEGArray()
    nwbfile = recording.data
    electrodes_table = nwbfile.electrodes.to_dataframe()
    locations = electrodes_table["location"]
    ch_names = [f"{locations[i]}_{i}" for i in range(len(locations))]
    if show_bad:
        bad_chans = list(electrodes_table[electrodes_table["clean"] == "Outlier"].index)
        bad_chans = [ch_names[i] for i in bad_chans]
    else:
        bad_chans = []
    sr = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    lfp_data = nwbfile.processing[type_]["LFP"]["ElectricalSeries"].data[:].T
    for sig in lfp_data:
        eeg = smr.EEG.from_numpy(sig, sr)
        eeg.conversion = 0.0000001
        eeg_array.append(eeg)
    try:
        average_signal = nwbfile.processing["average_lfp"]
        names = []
        for k in average_signal.data_interfaces:
            sig = average_signal[k].data[:]
            eeg = smr.EEG.from_numpy(sig, sr)
            eeg.conversion = 0.0000001
            eeg_array.append(eeg)
            names.append(k)
        ch_names.extend(names)
    except KeyError:
        print("WARNING: No average signal found")

    fig = eeg_array.plot(
        ch_names=ch_names,
        bad_chans=[str(i) for i in bad_chans],
        title=recording.get_name_for_save(),
        show=False,
    )

    fig.savefig(output_path, dpi=100)
    plt.close(fig)


plot_all_signals(recording, "test.png", False, False)

# %%
def detect_outlying_signals(signals, z_threshold=1.1):
    """
    Detect signals that are outliers from the average.

    Parameters
    ----------
    signals : np.ndarray
        Assumed to be an N_chans * N_samples iterable.
    z_threshold : float
        The threshold for the mean signal z-score to be an outlier.

    Returns
    -------
    good : np.ndarray
        The clean signals
    outliers : np.ndarray
        The outliers
    good_idx : list
        The indices of the good signals
    outliers_idx : list
        The indices of the bad signals
    z_scores : np.ndarray
        The array of z-scores.

    """
    avg_sig = np.mean(signals, axis=0)
    std_sig = np.std(signals, axis=0)
    std_sig = np.where(std_sig == 0, 1, std_sig)
    z_scores, good, bad = _split_signals_by_zscore(
        signals, z_threshold, avg_sig, std_sig
    )
    good_signals = np.array([signals[i] for i in good])
    bad_signals = np.array([signals[i] for i in bad])

    return good_signals, bad_signals, good, bad, z_scores


def _split_signals_by_zscore(signals, z_threshold, avg_sig, std_sig):
    """Split signals into those with z_scores above/below the z_threshold."""
    z_scores = np.zeros(shape=(len(signals), len(signals[0])))
    for i, s in enumerate(signals):
        if np.sum(np.abs(s)) < 0.2:
            z_scores[i] = np.zeros(shape=len(s))
        else:
            z_scores[i] = np.abs((s - avg_sig) / std_sig)
    z_score_means = np.nanmean(z_scores, axis=1)
    z_threshold = z_threshold * np.median(z_score_means[z_score_means != 0])

    good, bad = [], []
    for i, val in enumerate(z_score_means):
        if val > z_threshold:
            bad.append(i)
        elif np.sum(np.abs(signals[i])) < 0.2:
            bad.append(i)
        else:
            good.append(i)
    if not good:
        raise RuntimeError(f"No good signals found, bad were {bad}")

    print(f"Excluded {len(bad)} signals with indices {bad}")
    return z_scores, good, bad


nwbfile = recording.data
lfp_data = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].data[:].T
res = detect_outlying_signals(lfp_data, z_threshold=1.2)

# %%
plt.show()

# %% Temp test
average_signals = recording.data.processing["average_lfp"]
fields = average_signals.data_interfaces.keys()
for f in sorted(itertools.combinations(fields, 2)):
    x = average_signals[f[0]].data[:]
    y = average_signals[f[1]].data[:]
    fs = average_signals[f[0]].rate
    print(fs)
    key = f"{f[0][:-4]}_{f[1][:-4]}"
    print(key)

# %%
l = []
nwbfile = recording.data
print(recording.source_file)
coherence_df = nwbfile.processing["lfp_coherence"]["coherence_table"].to_dataframe()
print(coherence_df.columns)
region = coherence_df["label"].values[0]
group = "balh"
l.extend(
    [group, region, f_val, c_val]
    for f_val, c_val in zip(
        coherence_df["frequency"].values[0], coherence_df["coherence"].values[0]
    )
)

l

# %%
def grab_psds(nwbfile):
    psd_table = nwbfile.processing["lfp_power"]["power_spectra"].to_dataframe()
    electrodes_table = nwbfile.electrodes.to_dataframe()

    return psd_table, electrodes_table


def split_psds(psd_table, electrodes_table):
    normal_psds = psd_table[:-2][electrodes_table["clean"] == "Normal"]
    outlier_psds = psd_table[:-2][electrodes_table["clean"] == "Outlier"]

    return normal_psds, outlier_psds


nwbfile = recording.data
psd_table, electrodes_table = grab_psds(nwbfile)

# %%
normal_psds, outlier_psds = split_psds(psd_table, electrodes_table)

regions = ["SUB", "RSC"]
l = []


def add_psds_for_region_to_list(l, normal_psds, outlier_psds, region):
    clean_psds_in_region = normal_psds[normal_psds["region"] == region]
    outlier_psds_in_region = outlier_psds[outlier_psds["region"] == region]
    average_psd_for_clean = np.mean(clean_psds_in_region["power"], axis=0)
    average_psd_for_outlier = np.mean(outlier_psds_in_region["power"], axis=0)
    l.extend(
        [x, y, "Clean", region]
        for (x, y) in zip(average_psd_for_clean, normal_psds.iloc[0]["frequency"])
    )
    if len(outlier_psds_in_region) != 0:
        l.extend(
            [x, y, "Outlier", region]
            for (x, y) in zip(average_psd_for_outlier, normal_psds.iloc[0]["frequency"])
        )


for region in regions:
    add_psds_for_region_to_list(l, normal_psds, outlier_psds, region)

headers = ["Power (Db)", "Frequency (Hz)", "Type", "Brain Region"]
psd_dataframe = list_to_df(l, headers=headers)

# %%

max_frequency = 30

smr.set_plot_style()
sns.lineplot(
    x="Frequency (Hz)",
    y="Power (Db)",
    style="Type",
    hue="Brain Region",
    data=psd_dataframe[psd_dataframe["Frequency (Hz)"] < max_frequency],
)

# %% Average plots
l = []
headers = ["Power (Db)", "Frequency (Hz)", "Brain Region"]
for r in regions:
    psd = psd_table.loc[psd_table["label"] == f"{r}_avg"]
    l.extend(
        [x, y, r] for x, y in zip(psd["power"].array[0], psd["frequency"].array[0])
    )
avg_df = list_to_df(l, headers=headers)
sns.lineplot(
    x="Frequency (Hz)",
    y="Power (Db)",
    hue="Brain Region",
    data=avg_df[avg_df["Frequency (Hz)"] < max_frequency],
)

# %%
eeg_array = smr.EEGArray()
sr = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
lfp_data = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].data[:].T
for sig in lfp_data:
    eeg = smr.EEG.from_numpy(sig, sr)
    eeg.conversion = 0.001  # mV
    eeg_array.append(eeg)

bad_chans = electrodes_table[electrodes_table["clean"] == "Outlier"].index
eeg_array.plot(
    ch_names=[str(i) for i in range(len(eeg_array))],
    bad_chans=[str(i) for i in bad_chans],
    title=recording.get_name_for_save(),
)

# %%
