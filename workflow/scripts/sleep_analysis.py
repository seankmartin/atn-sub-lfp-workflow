from operator import sub

import numpy as np
import pandas as pd
import simuran as smr
import yasa
from ripple_detection import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    filter_ripple_band,
)
from simuran.bridges.mne_bridge import convert_signals_to_mne

from sleep_utils import create_events, mark_rest


def main(input_path, out_dir):
    df = pd.read_csv(input_path, parse_dates=["date_time"])
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    plot_recordings_per_animal(sleep, out_dir / "rat_stats.png")


def find_rest_periods(r):
    nwbfile = r.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    speed_rate = np.mean(np.diff(speed))
    # TODO add raw EGF for these purposes
    lfp_rate = r.processing["egf"].rate
    lfp = r.processing["lfp_egf"]
    resting_array = mark_rest(speed, lfp, lfp_rate, speed_rate)
    on_target = r.attrs["RSC on target"]
    # TODO this actually not work for EGF - should do per channel instead
    sub_signal = nwbfile.processing["average_lfp"]["SUB_avg"].data[:]
    rsc_signal = nwbfile.processing["average_lfp"]["RSC_avg"].data[:]
    fs = nwbfile.processing["average_lfp"]["SUB_avg"].rate
    convert_to_mne(sub_signal, rsc_signal, resting_array, on_target, fs)
    detect_spindles(sub_signal, rsc_signal)


def convert_to_mne(sub_signal, rsc_signal, events, on_target, fs=250):
    sub_eeg = smr.Eeg.from_numpy(sub_signal, fs=fs)
    signals = [sub_eeg]
    if on_target:
        rsc_eeg = smr.Eeg.from_numpy(rsc_signal, fs=fs)
        signals.append(rsc_eeg)

    record = convert_signals_to_mne(signals)
    events = np.array[[events] * len(signals)]
    return create_events(record, events)


def detect_spindles(mne_data):
    """See https://github.com/raphaelvallat/yasa/tree/master/notebooks

    For demos.
    """
    sub_signal = sub_signal.data * sub_signal.conversion * 10 ^ 6
    rsc_signal = rsc_signal.data * rsc_signal.conversion * 10 ^ 6
    data = np.array[[sub_signal, rsc_signal]]

    sp = yasa.spindles_detect(
        mne_data,
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
        freq_sp=(12, 15),
        multi_only=True,
        verbose="error",
    )

    return sp


def detect_ripples(sub_signal, rsc_signal):
    SAMPLING_FREQUENCY = 250.0
    time = np.asarray([i * 0.004 for i in range(0, len(lfp))])
    filtered_lfps = filter_ripple_band(lfps)
    ripple_times = Kay_ripple_detector(
        time,
        filtered_lfps,
        mov,
        SAMPLING_FREQUENCY,
        speed_threshold=2.5,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        smoothing_sigma=0.004,
        close_ripple_threshold=0.1,
    )
    ripple_times.head()
    true_ripple_midtime = [0.324, 1.42]
    RIPPLE_DURATION = 0.100
    Karlsson_ripple_times = Karlsson_ripple_detector(
        time, filtered_lfps, mov, SAMPLING_FREQUENCY
    )
    true_ripple_midtime = [3.80, 5.9]

    RIPPLE_DURATION = 0.100
    fig, ax = plt.subplots(figsize=(15, 3))
    plt.plot(time[500:1500], lfps[500:1500, :])

    for midtime in true_ripple_midtime:
        plt.axvspan(
            midtime - RIPPLE_DURATION / 2,
            midtime + RIPPLE_DURATION / 2,
            alpha=0.3,
            color="green",
            zorder=10 - 0,
        )
        Kay_ripple_times = Kay_ripple_detector(
            time, filtered_lfps, mov, SAMPLING_FREQUENCY
        )


if __name__ == "__main__":
    main(input_path, out_dir)
