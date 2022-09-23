# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

# %% Grab recording
loader = smr.loader("nwb")
path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed\CSR6--screening_small sq--22022018--22022018_CSR6_screening_small sq_1.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()
nwbfile = recording.data
config = smr.config_from_file(parent.parent / "config" / "simuran_params.yml")

# %% Grab data from recording
from frequency_analysis import calculate_psd
from lfp_clean import LFPAverageCombiner, NWBSignalSeries
from scipy.integrate import simps


def bp(f_norm, Pxx_norm):
    idx_band = np.logical_and(f_norm >= 6, f_norm <= 12)
    total_power1 = simps(Pxx_norm, f_norm)
    abs_power1 = simps(Pxx_norm[idx_band], f_norm[idx_band])
    return abs_power1 / total_power1


lfp_data = nwbfile.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
electrodes_table = nwbfile.electrodes.to_dataframe()
average_signal = nwbfile.processing["average_lfp"]

# %% Calculate the issue with average signal in NWB
results = []

for i, data in enumerate(lfp_data):
    row = electrodes_table.iloc[i]
    f, Pxx, _ = calculate_psd(data, scale="decibels")
    results.append([row.label, bp(f, Pxx), row.clean, row.location])

for r in results:
    print(r)

a_list = [r[1] for r in results if r[-1] == "SUB"]
avg = np.mean(a_list)

sig = average_signal["SUB_avg"].data[:]
f, Pxx, _ = calculate_psd(sig, scale="decibels")
r2 = bp(f, Pxx)
print(avg, r2)

s = 3000
ss = NWBSignalSeries(recording, normalised=True)
ss.filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
sub_ss = ss.select_electrodes("group_name", ["BE0", "BE1"])
combiner = LFPAverageCombiner(
    z_threshold=config["z_score_threshold"],
    remove_outliers=True,
    z_normalise=False,
)

res = combiner.combine(sub_ss)

# %% Plot the issue with the average signal in NWB
avg_res = res["SUB"]["average_signal"]
fig, axes = plt.subplots(5, 1)
print(np.std(lfp_data[0]))
avg = np.mean(np.array([lfp_data[0][:s] + lfp_data[1][:s]]), axis=0)
print("Plotting")
axes[0].plot(lfp_data[0][:s], c="k")
axes[1].plot(lfp_data[1][:s], c="k")
axes[2].plot(average_signal["SUB_avg"].data[:s], c="k")
axes[3].plot(avg_res[:s], c="k")
axes[4].plot(avg[:s], c="k")
plt.show()
