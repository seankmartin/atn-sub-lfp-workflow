from operator import sub

import numpy as np
import pandas as pd
import simuran as smr
import yasa
from ripple_detection import Kay_ripple_detector, filter_ripple_band
from simuran.bridges.mne_bridge import convert_signals_to_mne

from sleep_utils import (
    create_events,
    ensure_sleeping,
    mark_rest,
    plot_recordings_per_animal,
    spindles_exclude_resting,
)


def main(input_path, out_dir):
    df = pd.read_csv(input_path, parse_dates=["date_time"])
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    plot_recordings_per_animal(sleep, out_dir / "rat_stats.png")
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(sleep, loader)
    for r in rc.load_iter():
        if not ensure_sleeping(r):
            print(f"Too much movement in {r.source_file} for sleep")
        long_resting = spindle_control(r)
        ripple_control(r, long_resting)


def spindle_control(r):
    resting_array, long_resting = find_resting(r)
    mne_array = convert_to_mne(r, resting_array)
    sp = detect_spindles(mne_array)
    print(sp.summary())
    breakpoint()
    if sp is not None:
        spindles, rest = spindles_exclude_resting(
            sp.summary(), resting_array, mne_array, True
        )
    breakpoint()
    print(spindles, rest)
    return long_resting


def find_resting(r):
    nwbfile = r.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    speed_rate = np.mean(np.diff(speed))
    sub_signal = r.processing["average_lfp"]["SUB_avg"].data[:]
    sub_rate = r.processing["average_lfp"]["SUB_avg"].rate
    rate2 = r.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"].rate
    return mark_rest(speed, sub_signal, sub_rate, rate2, speed_rate)


def convert_to_mne(r, events):
    lfp_egf = r.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp_egf.rate
    lfp_data = lfp_egf.data[:]
    on_target = r.attrs["RSC on target"]
    electrodes = r.electrodes.to_dataframe()
    signal_array = [
        smr.Eeg.from_numpy(lfp, lfp_rate)
        for i, lfp in enumerate(lfp_data)
        if on_target or electrodes["region"][i] != "RSC"
    ]

    bad_chans = list(electrodes["clean"])
    ch_names = [
        f"{name}_{i}"
        for i, name in enumerate(electrodes["region"])
        if on_target or electrodes["region"][i] != "RSC"
    ]
    mne_array = convert_signals_to_mne(signal_array, ch_names, bad_chans)
    events = np.array[[events] * len(signal_array)]
    return create_events(mne_array, events)


def detect_spindles(mne_data):
    """See https://github.com/raphaelvallat/yasa/tree/master/notebooks

    For demos.
    """
    return yasa.spindles_detect(
        mne_data,
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
        freq_sp=(12, 15),
        verbose="error",
    )


def ripple_control(r, long_resting):
    lfp_egf = r.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp_egf.rate
    lfp_data = lfp_egf.data[:]
    filtered_lfps = filter_ripple_band(lfp_data)
    time = [i / lfp_rate for i in range(len(filtered_lfps[0]))]
    speed = r.processing["behaviour"]["running_speed"].data[:]
    speed_long = [speed[i // lfp_rate] for i in range(lfp_data[0])]
    # TODO detect movement ourselves
    ripple_times = Kay_ripple_detector(
        time,
        filtered_lfps,
        speed_long,
        lfp_rate,
        speed_threshold=2.5,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        smoothing_sigma=0.004,
        close_ripple_threshold=0.1,
    )
    return ripple_times


if __name__ == "__main__":
    main(input_path, out_dir)
