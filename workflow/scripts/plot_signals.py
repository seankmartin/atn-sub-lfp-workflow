import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr


def plot_all_signals(recording, output_path):
    eeg_array = smr.EEGArray()
    nwbfile = recording.data
    electrodes_table = nwbfile.electrodes.to_dataframe()
    sr = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    lfp_data = nwbfile.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
    for sig in lfp_data:
        eeg = smr.EEG.from_numpy(sig, sr)
        eeg.conversion = 0.001  # mV
        eeg_array.append(eeg)

    bad_chans = electrodes_table[electrodes_table["clean"] == "Outlier"].index
    fig = eeg_array.plot(
        ch_names=[str(i) for i in range(len(eeg_array))],
        bad_chans=[str(i) for i in bad_chans],
        title=recording.get_name_for_save(),
        show=False,
    )
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


plot_all_signals(recording, path)
