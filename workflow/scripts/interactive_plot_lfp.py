# %%
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
path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed_nwbfiles\CSR6--screening_small sq--20032018--20032018_CSR6_screening_small sq_8.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()

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
