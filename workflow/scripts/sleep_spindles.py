import sys

sys.path.insert(0, "D:/Beths/")
# import re
import math
import os
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pingouin as pg
import yasa
from bandPower import *

## Import from my files
from data_lfp import load_lfp_Axona, mne_lfp_Axona
from data_pos import RecPos
from mne.filter import filter_data, resample
from pandas_profiling import ProfileReport
from scipy import signal
from tqdm import tqdm

warnings.filterwarnings("ignore")


def open_sleep_files():
    df = pd.read_csv("data_scheme.csv")
    sleep = df.loc[df.sleep == 1]
    sleep_files = (
        df.loc[df.sleep == 1, ["folder", "filename"]].agg("/".join, axis=1).values
    )
    sleep_files = [file for file in sleep_files if "awake" not in file.split("_")]
    sleep_files = [file for file in sleep_files if "awake.set" not in file.split("_")]
    filenames = [r.strip().split("/")[-1] for r in sleep_files]
    sleep = df[df.filename.isin(filenames)]
    sleep_files = (
        sleep.loc[sleep.sleep == 1, ["folder", "filename"]].agg("/".join, axis=1).values
    )
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
    plt.show()

    ax = (
        sleep["treatment"]
        .value_counts()
        .plot(kind="bar", figsize=(9, 5), title="Treatment numberd in sleep animals.")
    )
    ax.set_xlabel("Treatment type")

    ax.set_ylabel("Frequency of recordings")
    plt.show()


def mark_resting(
    file, mne_data, tresh=2.5, window_sec=2
):  # Mark timestamps were the speed > 2.5 cm/s
    """Returns ones for the time windows where the animal was moving with a speed smaller than treshold
    Inputs:
        file(str): filename to be analysed
        tresh(float): speed treshold in cm/s
        window_sec(int): Window in seconds to define resting epochs
    Returns:
        resting(arr): 1 (ones) for resting 0 (zeros) for movement
    """
    #     cnt = 0 # DEBUG

    # Calculate movement based on speed treshold
    file_resting = []
    pos = RecPos(file)

    #     mne_data = mne_lfp_Axona(file)
    window = window_sec * 250
    speed = pos.get_speed()
    moving = np.zeros(len(speed) * 5)

    for i in range(0, len(speed)):
        if speed[i] > tresh:
            moving[5 * i : 5 * i + 5] = 1

    for data in mne_data.get_data():
        result = np.zeros(len(data))
        for i in range(0, len(data) - window, window // 2):
            bp = bandpower_ratio(data[i : i + window], window_sec)
            if (
                sum(moving[i : i + window]) == 0 and bp < 2
            ):  # running speed < 2.5cm/s , and theta/delta power ratio < 2
                result[i : i + window] = 1  # ones for resting periods
        #                 cnt+=1 # DEBUG
        file_resting.append(result)
    #     print(f'Total epochs for window = {window_sec}s and tresh = {tresh}cm/s is: {cnt} epochs') # DEBUG
    return np.asarray(file_resting)


def create_events(record, events):
    """Create events on MNE object
    Inputs:
        record(mne_object): recording to add events
        events_time(2D np array): array 0,1 with same lenght of recording dimension (1, lengt(record))
    output:
    record(mne_object): Record with events added
    """
    events = np.reshape(events, (1, -1))
    try:
        assert len(record.times) == events.shape[1]
        stim_data = events
        info = mne.create_info(["STI"], record.info["sfreq"], ["stim"])
        stim_raw = mne.io.RawArray(stim_data, info)
        record.add_channels([stim_raw], force_update_info=True)
    except AssertionError as error:
        print(error)
        print("The lenght of events needs to be equal to record lenght.")
    return record


def process_spindles(mne_data):
    """"""
    sp = yasa.spindles_detect(
        mne_data,
        sf=250,
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
        freq_sp=(12, 15),
        multi_only=True,
        verbose="error",
    )
    try:
        df = sp.summary()
    except:
        return None
    return df


def spindles_exclude_resting(mne_data, resting, ch_list=None, out_rest=False):
    #     resting_df = pd.DataFrame(resting.T, columns = ['ch_'+str(1+i) for i in range(0, len(mne_data.info['ch_names']))])
    resting_df = pd.DataFrame(resting.T, columns=mne_data.info["ch_names"])
    resting_df["time"] = [i * 0.004 for i in range(0, len(mne_data))]
    resting_df = resting_df.set_index("time")
    spindles_df = process_spindles(mne_data)
    cnt = 0
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


def divide_files_into_treatment_groups(sleep):
    musc = sleep.loc[sleep.treatment == "muscimol"]
    control = sleep.loc[sleep.treatment == "Control"]
    lesion = sleep.loc[sleep.treatment == "lesion"]
    print(f"Muscimol: {len(musc)} recordings in {len(musc.rat.unique())} animals")
    print(f"Control: {len(control)} recordings in {len(control.rat.unique())} animals")
    print(f"Lesion: {len(lesion)} recordings in {len(lesion.rat.unique())} animals")
    msc_files = (
        sleep.loc[sleep.treatment == "muscimol", ["folder", "filename"]]
        .agg("/".join, axis=1)
        .values
    )
    cnt_files = (
        sleep.loc[sleep.treatment == "Control", ["folder", "filename"]]
        .agg("/".join, axis=1)
        .values
    )
    les_files = (
        sleep.loc[sleep.treatment == "lesion", ["folder", "filename"]]
        .agg("/".join, axis=1)
        .values
    )


def possible_conversion():
    import simuran

    # A. Get the parameters
    mapping_directory = r"E:\Repos\lfp_atn\lfp_atn_simuran\recording_mappings"

    # Option 1. Add to DF
    def animal_to_mapping(s):
        cl_13 = "CL-SR_1-3.py"
        cl_46 = "CL-SR_4-6.py"
        d = {
            "CSubRet1": cl_13,
            "CSubRet2": cl_13,
            "CSubRet3": cl_13,
            "CSubRet4": cl_46,
            "CSubRet5": cl_46,
            "CSR6": cl_46,
        }

        return d.get(s, "NOT_EXIST")

    sleep["mapping"] = sleep.rat.apply(animal_to_mapping)

    # Option 2. Add it when making the big CSV

    # B. Get a recording container object (load data)
    beths_data_folder = r"D:\SubRet_recordings_imaging"

    # Set load to false if lots of data
    recording_container = simuran.recording_container_from_df(
        df, beths_data_folder, mapping_directory, load=True
    )

    # B1. How to clean / select channels
    from lfp_atn_simuran.analysis import LFPClean
    from lfp_atn_simuran.analysis.parse_cfg import parse_cfg_info

    config = parse_cfg_info()
    cleaning_method = config["clean_method"]
    clean_kwargs = config["clean_kwargs"]
    lc = LFPClean(method=cleaning_method, visualise=False, show_vis=True)

    # C. Converting the data to what you have
    for i in range(len(recording_container)):
        recording = recording_container[i]

        # Here is where the cleaning / selecting happens
        fmin = 1
        fmax = 100

        # signals_grouped_by_region is a simple dictionary
        # keys are regions, and values a single LFP for that region
        signals_grouped_by_region = lc.clean(
            recording.signals, fmin, fmax, method_kwargs=clean_kwargs
        )["signals"]

        # A loop like this is usual
        for region, signal in signals_grouped_by_region.items():
            # Currently signal is simuran.EEG
            # for example signal.samples
            results[region] = analysis_value

        result = new_mark_resting(recording)

        recording_container[i].results = result

    # Save all the results
    print(recording_container.results)
    # recording_container.save_summary_data(
    #     output_file_location, what_to_save)

    def new_mark_resting(
        recording, tresh=2.5, window_sec=2
    ):  # Mark timestamps were the speed > 2.5 cm/s
        """Returns ones for the time windows where the animal was moving with a speed smaller than treshold
        Inputs:
            file(str): filename to be analysed
            tresh(float): speed treshold in cm/s
            window_sec(int): Window in seconds to define resting epochs
        Returns:
            resting(arr): 1 (ones) for resting 0 (zeros) for movement
        """
        #     cnt = 0 # DEBUG
        speed = recording.spatial.speed  # ideal
        # if gives out about units do speed.values (check astropy)

        # speed = recording.spatial.underlying.speed # alternative

        config = parse_cfg_info()
        cleaning_method = config["clean_method"]
        clean_kwargs = config["clean_kwargs"]
        lc = LFPClean(method=cleaning_method, visualise=False, show_vis=True)
        fmin = 1
        fmax = 100
        # signals_grouped_by_region is a simple dictionary
        # keys are regions, and values a single LFP for that region
        signals_grouped_by_region = lc.clean(
            recording.signals, fmin, fmax, method_kwargs=clean_kwargs
        )["signals"]

        # Calculate movement based on speed treshold
        file_resting = []

        #     mne_data = mne_lfp_Axona(file)
        window = window_sec * 250
        speed = pos.get_speed()
        moving = np.zeros(len(speed) * 5)

        for i in range(0, len(speed)):
            if speed[i] > tresh:
                moving[5 * i : 5 * i + 5] = 1

        # New
        results = {}
        for region, signal in signals_grouped_by_region.items():
            data = signal.samples
            res = bandpower_ratio(data[i : i + window], window_sec)
            results[f"{region}__bandpower"] = res

        ## Old
        for data in mne_data.get_data():
            result = np.zeros(len(data))
            for i in range(0, len(data) - window, window // 2):
                bp = bandpower_ratio(data[i : i + window], window_sec)
                if (
                    sum(moving[i : i + window]) == 0 and bp < 2
                ):  # running speed < 2.5cm/s , and theta/delta power ratio < 2
                    result[i : i + window] = 1  # ones for resting periods
            #                 cnt+=1 # DEBUG
            file_resting.append(result)
        #     print(f'Total epochs for window = {window_sec}s and tresh = {tresh}cm/s is: {cnt} epochs') # DEBUG
        return np.asarray(file_resting)


def ripples():
    from ripple_detection import (
        Karlsson_ripple_detector,
        Kay_ripple_detector,
        filter_ripple_band,
    )
    from ripple_detection.simulate import simulate_time

    test_file = "/mnt/d/Beths/CanCSCa1/smallsq_sleep/10082018/s3_sleep/10082018_CanCSubCa1_smallsq_sleep_1_3.set"
    mne_data = mne_lfp_Axona(test_file)
    pos = RecPos(test_file)
    speed = pos.get_speed()
    lfps = mne_data.get_data(["ch_1", "ch_20"]).T
    mov = np.zeros(lfps.shape[0])
    for i, sp in enumerate(speed):  # Speed to be on the same lenght as lfp time
        mov[5 * i : 5 * i + 5] = sp
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


def ensure_sleeping():
    true_sleep = []
    for file in sleep_files:
        pos = RecPos(file)
        x, y = pos.get_position()
        resting = 1 - mark_moving(file, tresh=1.0)  # 1-moving
        if (100 * (sum(resting)) / len(resting)) > 25:
            true_sleep.append(file.strip())
    true_sleeps = [r.strip().split("/")[-1] for r in true_sleep]
    sleep_files = df[df.filename.isin(true_sleeps)]
