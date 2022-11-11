import ast
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
import simuran as smr
from scipy.signal import decimate
from simuran.bridges.mne_bridge import convert_signals_to_mne
from skm_pyutils.table import df_from_file

here = Path(__file__).parent
sleep_dir = here.parent.parent / "results" / "sleep"
filename = r"results\processed\CSubRet5_sham--recording--sleep--08122017--S2 sleep--08122017_CSR5_sleep_2_2_sleep.nwb"

spindles_df = df_from_file(sleep_dir / "spindles.csv")
ripples_df = df_from_file(sleep_dir / "ripples.csv")

DATA_LEN = 200


def add_ripples_annotation(mne_data, ripples_df, filename):
    annotations_info = ([], [], [])
    ripples = ripples_df[ripples_df["Filename"] == filename]

    for i, row in ripples.iterrows():
        times = ast.literal_eval(row["Ripple Times"])
        region = row["Brain Region"]
        detector = row["Detector"]
        annotations_info[0].extend((t[0] for t in times if t[1] <= DATA_LEN))
        annotations_info[1].extend((t[1] - t[0] for t in times if t[1] <= DATA_LEN))
        annotations_info[2].extend(
            (f"{region}_r_{detector}") for t in times if t[1] <= DATA_LEN
        )

    annotations = mne.Annotations(*annotations_info)
    mne_data.set_annotations(annotations)


def add_spindles_annotation(mne_data, spindles_df, filename):
    annotations_info = ([], [], [])
    spindles = spindles_df[spindles_df["Filename"] == filename]

    for i, row in spindles.iterrows():
        times = ast.literal_eval(row["Spindle Times"])
        region = row["Brain Region"]
        annotations_info[0].extend((t[0] for t in times))
        annotations_info[1].extend((t[1] - t[0] for t in times))
        annotations_info[2].extend((f"{region}_s") for t in times)

    annotations = mne.Annotations(*annotations_info)
    mne_data.set_annotations(annotations)

    return annotations_info[0][0]


def convert_to_mne(r):
    nwbfile = r.data
    lfp = nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp.rate
    lfp_data = lfp.data[: int(lfp_rate * DATA_LEN)].T
    electrodes = nwbfile.electrodes.to_dataframe()
    signal_array = [smr.Eeg.from_numpy(lfp, lfp_rate) for lfp in lfp_data]

    bad_chans = list(electrodes["clean"])
    ch_names = [f"{name}_{i}" for i, name in enumerate(electrodes["location"])]
    return convert_signals_to_mne(signal_array, ch_names, bad_chans)


def convert_low_rate_to_mne(r):
    nwbfile = r.data
    lfp = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp.rate
    lfp_data = lfp.data[:].T
    electrodes = nwbfile.electrodes.to_dataframe()
    signal_array = [smr.Eeg.from_numpy(lfp, lfp_rate) for lfp in lfp_data]

    bad_chans = list(electrodes["clean"])
    ch_names = [f"{name}_{i}" for i, name in enumerate(electrodes["location"])]
    return convert_signals_to_mne(signal_array, ch_names, bad_chans)


def check_decimation(signals):
    filtered_signals = mne.filter.filter_data(signals, 4800, 150, 250)
    fig, axes = plt.subplots(2 * len(signals))
    x = [i / 4800 for i in range(0, 4800 * 20)]
    for i, s in enumerate(filtered_signals):
        axes[i].plot(x, s[: 4800 * 20], c="k")
    x = [i / 1600 for i in range(0, 1600 * 20)]
    filtered_lfps = decimate(filtered_signals, 3, zero_phase=True, axis=-1)
    for i, s in enumerate(filtered_lfps):
        axes[2 + i].plot(x, s[: 1600 * 20], c="b")


loader = smr.loader_from_string("nwb")
recording = smr.Recording(source_file=filename, loader=loader)
recording.load()
nwbfile = recording.data


def plot_decimation(nwbfile):
    signals = (
        nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"].data[:, :2].T
    )
    check_decimation(signals)
    plt.show()


def ripples_plot(recording, ripples_df, filename):
    mne_data = convert_to_mne(recording)
    add_ripples_annotation(mne_data, ripples_df, filename)
    max_val = 1.8 * np.max(np.abs(mne_data.get_data(stop=DATA_LEN)))
    scalings = {"eeg": max_val}
    mne_data.plot(
        duration=6.0,
        n_channels=4,
        scalings=scalings,
        lowpass=150,
        highpass=250,
        show=True,
    )
    input("Press enter to continue...")


def spindles_plot(recording, spindles_df, filename):
    mne_data = convert_low_rate_to_mne(recording)
    first = add_spindles_annotation(mne_data, spindles_df, filename)
    max_val = 1.8 * np.max(np.abs(mne_data.get_data()))
    scalings = {"eeg": max_val}
    mne_data.plot(duration=6.0, n_channels=4, scalings=scalings, start=first, show=True)
    input("Press enter to continue...")


def speed_plot(recording, ripples_df, filename):
    speed = recording.data.processing["behavior"]["running_speed"].data[:]
    timestamps = recording.data.processing["behavior"]["running_speed"].timestamps[:]
    ripples = ripples_df[ripples_df["Filename"] == filename]

    print(ripples["Resting Times"].iloc[0])
    plt.plot(timestamps, speed)
    plt.show()
    input("Press Enter to continue")


speed_plot(recording, ripples_df, filename)
spindles_plot(recording, spindles_df, filename)
ripples_plot(recording, ripples_df, filename)
plot_decimation(nwbfile)
