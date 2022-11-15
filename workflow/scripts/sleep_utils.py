import logging

import matplotlib.pyplot as plt
import mne
import numpy as np
import simuran as smr
from simuran.bridges.neurochat_bridge import signal_to_neurochat

module_logger = logging.getLogger("simuran.custom.sleep_utils")


def mark_rest(speed, lfp, lfp_rate, speed_rate, tresh=2.5, window_sec=2, **kwargs):
    theta_min, theta_max = kwargs["theta_min"], kwargs["theta_max"]
    delta_min, delta_max = kwargs["delta_min"], kwargs["delta_max"]
    min_sleep_length = kwargs["min_sleep_length"]
    sleep_tol = kwargs["sleep_join_tol"]
    max_interval_size = kwargs["sleep_max_interval_size"]

    lfp_samples_per_speed = int(lfp_rate * speed_rate)
    moving = np.zeros(int(len(speed) * lfp_samples_per_speed))
    for i in range(len(speed)):
        if speed[i] > tresh:
            moving[lfp_samples_per_speed * i : lfp_samples_per_speed * (i + 1)] = 1

    window = int(window_sec * lfp_rate)
    half_window = window // 2
    result = np.zeros(len(lfp))

    for i in range(half_window, len(lfp), window):
        sig = smr.Eeg.from_numpy(lfp[i - half_window : i + half_window], lfp_rate)
        nc_sig = signal_to_neurochat(sig)
        bp = nc_sig.bandpower_ratio(
            [theta_min, theta_max], [delta_min, delta_max], window_sec
        )
        # running speed < 2.5cm/s , and theta/delta power ratio < 2
        if sum(moving[i - half_window : i + half_window]) == 0 and bp < 2:
            result[i - half_window : i + half_window] = 1

    intervaled = find_ranges(result, lfp_rate)
    module_logger.debug(f"Resting ranges {intervaled}")

    old_len = len(intervaled)
    new_intervals = combine_intervals(intervaled, tol=sleep_tol)
    new_len = len(new_intervals)
    while new_len != old_len:
        old_len = len(new_intervals)
        new_intervals = combine_intervals(new_intervals, tol=sleep_tol)
        new_len = len(new_intervals)

    module_logger.debug(f"Combined resting ranges to {new_intervals}")

    final_intervals = [
        val for val in new_intervals if (val[-1] - val[0]) > min_sleep_length
    ]
    module_logger.debug(f"Removed short intervals to {final_intervals}")

    final_intervals = cap_intervals(final_intervals, max_interval_size)
    module_logger.debug(f"Capped intervals at 200: {final_intervals}")

    return final_intervals, intervaled


def cap_intervals(intervals, tol=200):
    new_intervals = []
    for interval in intervals:
        start, end = interval
        while (len(new_intervals) == 0) or (new_intervals[-1][-1] != end):
            if (end - start) > (tol + 50):
                new_intervals.append((start, start + tol))
                start = start + tol
            else:
                new_intervals.append((start, end))
    return new_intervals


def combine_intervals(intervaled, tol=2):
    intervals = []
    i = 0
    while i < (len(intervaled) - 1):
        val = intervaled[i]
        if (intervaled[i + 1][0] - val[-1]) <= tol:
            intervals.append((val[0], intervaled[i + 1][-1]))
            i += 2
        else:
            intervals.append(val)
            i += 1

    if len(intervals) == 0:
        intervals.append(intervaled[-1])

    if intervals[-1][-1] != intervaled[-1][-1]:
        intervals.append(intervaled[-1])

    return intervals


def create_events(record, events):
    """Create events on MNE object
    Inputs:
        record(mne_object): recording to add events
        events_time(2D np array): array 0,1 with same lenght of recording dimension (1, lengt(record))
    output:
    record(mne_object): Record with events added
    """
    annotations = mne.Annotations(
        [t[0] for t in events], [t[1] - t[0] for t in events], "Resting"
    )
    record.set_annotations(annotations)
    return record


def mark_movement(speed, mne_array):
    n_channels = len(mne_array)
    resting = np.asarray(mark_rest(speed, tresh=2.5, mov=True))
    events = resting.reshape(n_channels, len(resting))
    return create_events(mne_array, events)


def spindles_exclude_resting(spindles_df, resting):
    times = []
    for channel in spindles_df.Channel.unique():
        sp_times = spindles_df.loc[spindles_df.Channel == channel][
            ["Start", "End"]
        ].values
        for t in sp_times:
            use_this_time = any(((t[0] >= r[0]) and (t[1] <= r[1]) for r in resting))
            if use_this_time:
                if not np.any(np.abs(np.array([t_[0] for t_ in times]) - t[0]) < 0.1):
                    times.append([t[0], t[1]])
    return times


def plot_recordings_per_animal(sleep, out_name):
    fig, axes = plt.subplots(2, 1)
    ax = axes[0]
    ax = (
        sleep["rat"]
        .value_counts()
        .plot(
            kind="bar",
            figsize=(9, 5),
            title="Number of sleep recordings for each animal.",
        )
    )
    ax.set_xlabel("Rat ID")
    ax.set_ylabel("Frequency of recordings")
    ax = axes[1]
    ax = (
        sleep["treatment"]
        .value_counts()
        .plot(kind="bar", figsize=(9, 5), title="Treatment number in sleep animals.")
    )
    ax.set_xlabel("Treatment type")
    ax.set_ylabel("Frequency of recordings")
    fig.savefig(out_name)
    plt.close(fig)


def ensure_sleeping(recording):
    nwbfile = recording.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    num_moving = np.count_nonzero(speed > 2.5)
    return (num_moving / len(speed)) < 0.5


def find_ranges(resting, srate):
    resting_times = []
    in_rest = False
    for i, val in enumerate(resting):
        if val and not in_rest:
            in_rest = True
            rest_start = i / srate
        elif not val and in_rest:
            in_rest = False
            rest_end = i / srate
            resting_times.append((rest_start, rest_end))
    if in_rest:
        resting_times.append((rest_start, ((len(resting) - 1) / srate)))
    return resting_times


def filter_ripple_band(data, srate, **kwargs):
    return mne.filter.filter_data(data, srate, 150, 250, **kwargs)
