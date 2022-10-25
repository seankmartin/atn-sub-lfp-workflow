import ast
from pathlib import Path

import mne
import numpy as np
import simuran as smr
from simuran.bridges.mne_bridge import convert_signals_to_mne
from skm_pyutils.table import df_from_file

here = Path(__file__).parent
sleep_dir = here.parent.parent / "results" / "sleep"
filename = r"d:\atn-sub-lfp-workflow\results\processed\CanCCaRet2--muscimol--08032019_muscimol_0pt2ulRandL--s10_smallsq_resting--08032019_CanCCaRet2_muscimol_smallsq_sleep_1_10.nwb"

spindles_df = df_from_file(sleep_dir / "spindles.csv")
ripples_df = df_from_file(sleep_dir / "ripples.csv")

DATA_LEN = 100


def add_annotation(mne_data, spindles_df, ripples_df, filename):
    annotations_info = ([], [], [])
    spindles = spindles_df[spindles_df["Filename"] == filename]
    ripples = ripples_df[ripples_df["Filename"] == filename]

    for i, row in spindles.iterrows():
        times = ast.literal_eval(row["Spindle Times"])
        region = row["Brain Region"]
        annotations_info[0].extend((t[0] for t in times if t[1] <= DATA_LEN))
        annotations_info[1].extend((t[1] - t[0] for t in times if t[1] <= DATA_LEN))
        annotations_info[2].extend((f"{region}_s") for t in times if t[1] <= DATA_LEN)

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


loader = smr.loader_from_string("nwb")
recording = smr.Recording(source_file=filename, loader=loader)
recording.load()
nwbfile = recording.data
mne_data = convert_to_mne(recording)
add_annotation(mne_data, spindles_df, ripples_df, filename)
max_val = 1.8 * np.max(np.abs(mne_data.get_data(stop=DATA_LEN)))
scalings = {"eeg": max_val}
fig = mne_data.plot(
    duration=6.0,
    n_channels=4,
    scalings=scalings,
    lowpass=150,
    highpass=250,
    show=True,
)
inp = input("Press enter to continue...")
