import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import simuran as smr
from simuran.bridges.neurochat_bridge import signal_to_neurochat


def mark_rest(
    speed, lfp, lfp_rate, lfp_rate2, speed_rate, tresh=2.5, window_sec=2, **kwargs
):
    """Returns ones for the time windows where the animal was moving with a speed smaller than treshold
    Inputs:
        file(str): filename to be analysed
        tresh(float): speed treshold in cm/s
        window_sec(int): Window in seconds to define resting epochs
    Returns:
        resting(arr): 1 (ones) for resting 0 (zeros) for movement
        There is a sample for each lfp sample
    """
    theta_min, theta_max = kwargs["theta_min"], kwargs["theta_max"]
    delta_min, delta_max = kwargs["delta_min"], kwargs["delta_max"]

    lfp_samples_per_speed = lfp_rate / speed_rate
    moving = np.zeros(len(speed) * lfp_samples_per_speed)
    for i in range(len(speed)):
        if speed[i] > tresh:
            moving[lfp_samples_per_speed * i : lfp_samples_per_speed * (i + 1)] = 1

    window = int(window_sec * lfp_rate)
    result = np.zeros(len(lfp))
    result2 = np.zeros(int(len(lfp) * (lfp_rate2 / lfp_rate)))
    for i in range(0, len(lfp) - window, window // 2):
        sig = smr.Eeg.from_numpy(lfp[i : i + window])
        nc_sig = signal_to_neurochat(sig)
        bp = nc_sig.bandpower_ratio(
            [theta_min, theta_max], [delta_min, delta_max], window_sec
        )
        # running speed < 2.5cm/s , and theta/delta power ratio < 2
        if sum(moving[i : i + window]) == 0 and bp < 2:
            result[i : i + window] = 1
            result2[i : i + (int(window_sec * lfp_rate2))] = 1
    return result, result2


def spindles_exclude_resting(mne_data, resting, ch_list=None, out_rest=False):
    resting_df = pd.DataFrame(resting.T, columns=mne_data.info["ch_names"])
    resting_df["time"] = [i * 0.004 for i in range(len(mne_data))]
    resting_df = resting_df.set_index("time")
    for channel in spindles_df.Channel.unique():
        sp_times = spindles_df.loc[spindles_df.Channel == channel][
            ["Start", "End"]
        ].values
        for time in sp_times:
            try:
                if sum(resting_df[channel][time[0] : time[1]]) <= 1:
                    spindles_df[
                        (spindles_df.Channel == channel)
                        & (spindles_df.Start == time[0])
                        & (spindles_df.End == time[1])
                    ] = np.nan
            except:
                print(f"Error in channel {channel}")
    if out_rest:
        return (
            spindles_df,
            resting_df,
        )  # Can be used to calculate proportion of spindles per time

    return spindles_df


def create_events(record, events):
    """Create events on MNE object
    Inputs:
        record(mne_object): recording to add events
        events_time(2D np array): array 0,1 with same lenght of recording dimension (1, lengt(record))
    output:
    record(mne_object): Record with events added
    """
    try:
        assert len(record.times) == events.shape[1]
        stim_data = events
        info = mne.create_info(["STI"], record.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(stim_data, info)
        record.add_channels([stim_raw], force_update_info=True)
    except AssertionError as error:
        print(error)
        print("The length of events needs to be equal to record length.")
    return record


def mark_movement(speed, mne_array):
    n_channels = len(mne_array)
    resting = np.asarray(mark_rest(speed, tresh=2.5, mov=True))
    events = resting.reshape(n_channels, len(resting))
    return create_events(mne_array, events)


def spindles_exclude_resting(spindles_df, resting, mne_data, out_rest=False):
    resting_df = pd.DataFrame(resting.T, columns=mne_data.info["ch_names"])
    resting_df["time"] = [i * 0.004 for i in range(len(mne_data))]
    resting_df = resting_df.set_index("time")
    for channel in spindles_df.Channel.unique():
        sp_times = spindles_df.loc[spindles_df.Channel == channel][
            ["Start", "End"]
        ].values
        for time in sp_times:
            try:
                if sum(resting_df[channel][time[0] : time[1]]) <= 1:
                    spindles_df[
                        (spindles_df.Channel == channel)
                        & (spindles_df.Start == time[0])
                        & (spindles_df.End == time[1])
                    ] = np.nan
            except BaseException:
                print(f"Error in channel {channel}")
    return (spindles_df, resting_df) if out_rest else spindles_df


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
    num_moving = len(np.nonzero(speed > 2.5))
    return num_moving / len(speed) > 0.25
