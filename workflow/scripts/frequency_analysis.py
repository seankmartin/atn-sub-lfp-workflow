import logging

import numpy as np
from scipy.signal import welch

module_logger = logging.getLogger("simuran.custom.frequency_analysis")


def calculate_psd(x, fs=250, fmin=1, fmax=100, scale="volts", warn=True):
    f, Pxx = welch(
        x * 0.001,
        fs=fs,
        nperseg=2 * fs,
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    f = f[np.nonzero((f >= fmin) & (f <= fmax))]
    Pxx = Pxx[np.nonzero((f >= fmin) & (f <= fmax))]
    Pxx_max = np.max(Pxx)
    if Pxx_max == 0:
        if warn:
            module_logger.warning("0-power found in LFP signal, directly returning")
        return (f, Pxx, Pxx_max)
    if scale == "decibels":
        Pxx = 10 * np.log10(Pxx / Pxx_max)
    elif scale != "volts":
        raise ValueError(f"Unsupported scale {scale}")

    return (f, Pxx, Pxx_max)
