import logging

import numpy as np
import simuran as smr

module_logger = logging.getLogger("simuran.custom.lfp_utils")


def detect_outlying_signals(signals, z_threshold=1.1):
    """
    Detect signals that are outliers from the average.

    Parameters
    ----------
    signals : np.ndarray
        Assumed to be an N_chans * N_samples iterable.
    z_threshold : float
        The threshold for the mean signal z-score to be an outlier.

    Returns
    -------
    good : np.ndarray
        The clean signals
    outliers : np.ndarray
        The outliers
    good_idx : list
        The indices of the good signals
    outliers_idx : list
        The indices of the bad signals
    z_scores : np.ndarray
        The array of z-scores.

    """
    avg_sig = np.mean(signals, axis=0)
    std_sig = np.std(signals, axis=0)
    std_sig = np.where(std_sig == 0, 1, std_sig)
    z_scores, good, bad = _split_signals_by_zscore(
        signals, z_threshold, avg_sig, std_sig
    )
    good_signals = np.array([signals[i] for i in good])
    bad_signals = np.array([signals[i] for i in bad])

    return good_signals, bad_signals, good, bad, z_scores


def _split_signals_by_zscore(signals, z_threshold, avg_sig, std_sig):
    """Split signals into those with z_scores above/below the z_threshold."""
    z_scores = np.zeros(shape=(len(signals), len(signals[0])))
    for i, s in enumerate(signals):
        if np.sum(np.abs(s)) < 0.2:
            z_scores[i] = np.zeros(shape=(len(s)))
        else:
            z_scores[i] = np.abs((s - avg_sig) / std_sig)
    z_score_means = np.nanmean(z_scores, axis=1)
    if np.all(z_score_means == 0):
        z_threshold = 10000
    else:
        z_threshold = z_threshold * np.median(z_score_means[z_score_means != 0])

    good, bad = [], []
    for i, val in enumerate(z_score_means):
        if val > z_threshold:
            bad.append(i)
        elif np.sum(np.abs(signals[i])) < 0.2:
            bad.append(i)
        else:
            good.append(i)
    if not good:
        module_logger.warning(f"No good signals found, bad were {bad} - returning all")
        good = bad
        bad = []

    module_logger.debug(f"Excluded {len(bad)} signals with indices {bad}")
    return z_scores, good, bad


def average_signals(signals):
    """
    Average a set of signals across the channel dimension.

    Parameters
    ----------
    signals : iterable
        Assumed to be an N_chans * N_samples iterable.

    Returns
    -------
    np.ndarray
        The averaged signal.

    """
    signals_ = np.array(signals) if type(signals) is not np.ndarray else signals
    return np.mean(signals_, axis=0)


def z_score_normalise_signals(signals, mode="mean"):
    """Z score signals, mode can be median instead of mean"""
    z_scored_signals = np.zeros_like(signals)
    for i in range(len(signals)):
        div = np.std(signals[i])
        div = np.where(div == 0, 1, div)
        avg = np.mean(signals[i]) if mode == "mean" else np.median(signals[i])
        z_scored_signals[i] = (signals[i] - avg) / div

    return z_scored_signals


def filter_sigs(signals, min_f, max_f, **filter_kwargs):
    eeg_array = smr.EegArray()
    for signal in signals:
        filt_s = signal.filter(min_f, max_f, inplace=False, **filter_kwargs)
        eeg_array.append(smr.Eeg(signal=filt_s))
    return eeg_array
