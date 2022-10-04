import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yasa

## Import from my files
from lib.data_lfp import load_lfp_Axona, mne_lfp_Axona
from lib.data_pos import RecPos
from mne.filter import filter_data, resample
from scipy import signal


def main(input_path, out_dir):
    df = pd.read_csv(input_path, parse_dates=["date_time"])
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    plot_recordings_per_animal(sleep, out_dir / "rat_stats.png")


def long_function_not_sure():
    import seaborn as sns

    sns.set(font_scale=1.2)
    t0 = 100
    tf = 130
    rec = record2.filter(l_freq=1, h_freq=30)
    data = rec[12][0][0][250 * t0 : 250 * tf]
    sfreq = 250
    print(len(data))
    # Define sampling frequency and time vector
    data = signal.resample(data, int(7500 / 5))  # resampling
    time = np.arange(len(data)) / sfreq
    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    plt.plot(time, data, lw=1.5, color="k")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage")
    # plt.xlim([time.min(), time.max()])
    plt.title(f"N3 sleep EEG data {tf-t0}s")
    sns.despine()
    info = mne.create_info(ch_names=["RSC"], ch_types=["eeg"], sfreq=250)
    ch1 = mne.io.RawArray(record1["ch_1"][0], info)
    # ch1.plot(show_scrollbars=False, show_scalebars=True, scalings=dict(eeg=100e-5))

    times = np.arange(len(ch1)) / 250.0

    # Plot the signal
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    plt.plot(times[12000:12800], record1["ch_1"][0][0][12000:12800], lw=1.5, color="k")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (uV)")
    # plt.xlim([times.min(), times.max()])
    plt.title("N2 sleep EEG data (2 spindles)")
    sns.despine()
    from scipy import signal

    # Define window length (4 seconds)
    win = 4 * sfreq
    freqs, psd = signal.welch(data, sfreq, nperseg=win)

    # Plot the power spectrum
    sns.set(font_scale=1.2, style="white")
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, psd, color="r", lw=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power spectral density (V^2 / Hz)")
    plt.ylim([0, psd.max() * 1.1])
    plt.title("Welch's periodogram")
    plt.xlim([0, freqs.max()])
    sns.despine()


if __name__ == "__main__":
    input_path = Path(__file__).parent.parent
    input_path = input_path / "subret_recordings.csv"
    main(input_path)
