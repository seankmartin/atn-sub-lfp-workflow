"""Clean LFP signals."""

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy

# 1. create object to be worked on - like this
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import simuran
from mne.preprocessing import ICA, read_ica

# TODO finish these ideas

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass
def SignalSeries(ABC):
    data: "np.ndarray"
    description: "pd.DataFrame"

    """
    Class for general ghostipy signal operations

    Attributes
    ----------
    data : np.ndarray
        The raw data of the signals. 
        Can be 1D, or 2D, where channels is dimension 1
    description : pd.DataFrame
        A row for each channel describing the signal.

    """

    @abstractmethod
    def group_by_brain_region(self):
        """Return signals grouped by the brain region."""


class LFPCombiner(ABC):
    """
    Used to combine LFP signals.

    For example, by averaging all LFP signals that fulfil a criteria.
    Or by performing ICA on all the channels for cleaning.
    """

    @abstractmethod
    def combine(self):
        """Combine LFP signals"""

    def vis_combining(self, result, signals, bad_chans=None, **kwargs):
        if self.method == "zscore":
            signals_ = deepcopy(signals)
            for i in range(len(signals_)):
                val = np.array(signals_[i].samples)
                div = np.std(val)
                div = np.where(div == 0, 1, div)
                mean = np.mean(val)
                signals_[i].samples = ((val - mean) / div) * signals_[i].samples.unit
        else:
            signals_ = signals

        if isinstance(result, dict):
            eeg_array = simuran.EegArray()
            _, eeg_idxs = signals_.group_by_property("channel_type", "eeg")
            eeg_sigs = signals_.subsample(idx_list=eeg_idxs, inplace=False)
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in eeg_sigs])

            for k, v in result.items():
                eeg_array.append(v)
        else:
            eeg_array = simuran.EegArray()
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in result])

        fig = eeg_array.plot(proj=False, show=self.show_vis, bad_chans=bad_chans)

        return fig


class LFPAverageCombiner(LFPCombiner):
    """Create an "average" signal for each brain region."""

    def combine(self):
        z_threshold = method_kwargs.get("z_threshold", 1.1)
        result, bad_chans = self.avg_method(
            signals,
            min_f,
            max_f,
            clean=True,
            z_threshold=z_threshold,
            **filter_kwargs,
        )

    def avg_method(
        self, signals, min_f, max_f, clean=True, z_threshold=1.1, **filter_kwargs
    ):
        lfp_signals = signals

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()

        bad_chans = []
        for region, (signals, _) in signals_grouped_by_region.items():
            val, bad_idx = average_signals(
                [s.samples for s in signals],
                z_threshold=z_threshold,
                verbose=True,
                clean=clean,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            if min_f is not None:
                eeg.filter(min_f, max_f, inplace=True, **filter_kwargs)
            output_dict[region] = eeg
            bad_chans += [signals[i].channel for i in bad_idx]

        return output_dict, bad_chans


class LFPICACombiner(LFPCombiner):
    """Perform ICA on the passed signals"""

    def combine(self):
        channels = method_kwargs.get("channels")
        prop = method_kwargs.get("pick_property", "channel")
        manual_ica = method_kwargs.get("manual_ica", False)
        if channels is not None:
            idxs = [
                i for i in range(len(signals)) if getattr(signals[i], prop) in channels
            ]
            bad_chans = [s.channel for s in signals if getattr(s, prop) not in channels]
            results["bad_channels"] = bad_chans
        else:
            idxs = None
        reconst, result, figs = self.ica_method(
            signals,
            chans_to_pick=idxs,
            ica_fname=ica_fname,
            manual=manual_ica,
            highpass=highpass,
        )
        results["cleaned"] = reconst
        results["ica_figs"] = figs

    def ica_method(
        self,
        signals,
        exclude=None,
        chans_to_pick=None,
        save=True,
        ica_fname=None,
        manual=True,
        highpass=0.0,
    ):
        skip_plots = not self.visualise
        show = self.show_vis
        if not isinstance(signals, simuran.EegArray):
            eeg_array = simuran.EegArray()
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in signals])

        regions = list(set([s.region for s in signals]))

        mne_array = signals.convert_signals_to_mne(verbose=False)
        mne_array.info["highpass"] = highpass

        loaded = False
        if save:
            home = os.path.abspath(os.path.expanduser("~"))
            default_save_location = os.path.join(home, ".skm_python", "ICA_files")
            os.makedirs(default_save_location, exist_ok=True)
            bit = "-auto"
            if manual:
                bit = ""
            if ica_fname is None:
                if os.path.basename(signals[0].source_file) != "<unknown>":
                    ica_fname = (
                        os.path.splitext(os.path.basename(signals[0].source_file))[0]
                        + f"{bit}-ica.fif.gz"
                    )
                else:
                    val = np.round(np.mean(signals[0].samples), 3)
                    name = str(val).replace(".", "-")
                    ica_fname = os.path.join(name) + f"{bit}-ica.fif.gz"
            elif not ica_fname.endswith("-ica.fif.gz"):
                if not ica_fname.endswith("-ica.fif"):
                    ica_fname = os.path.splitext(ica_fname)[0] + f"{bit}-ica.fif.gz"

            fname = os.path.join(default_save_location, ica_fname)

            if os.path.exists(fname):
                print("Loading ICA from {}".format(fname))
                ica = read_ica(fname, verbose="ERROR")
                loaded = True
                print(ica.exclude)

        if not loaded:
            ica = ICA(
                method="picard",
                random_state=42,
                max_iter="auto",
                fit_params=dict(ortho=True, extended=True),
                n_components=None,
            )

            ica.fit(mne_array)
            if manual:

                if exclude is None:
                    # Plot raw ICAs
                    ica.plot_sources(mne_array)

                    cont = input("Plot region overlay? (y|n) \n")
                    if cont.strip().lower() == "y":
                        reg_grps = []
                        for reg in regions:
                            temp_grp = []
                            for ch in mne_array.info.ch_names:
                                if reg in ch:
                                    temp_grp.append(ch)
                            reg_grps.append(temp_grp)
                        for grps in reg_grps:
                            ica.plot_overlay(
                                mne_array,
                                stop=int(30 * 250),
                                title="{}".format(grps[0][:3]),
                                picks=grps,
                            )
                else:
                    # ICAs to exclude
                    ica.exclude = exclude
                    if not skip_plots:
                        ica.plot_sources(mne_array)

            else:
                end_time = min(120.0, signals[0].get_end().value)
                ica = ica.detect_artifacts(
                    mne_array, start_find=20.0, stop_find=end_time
                )

        if save:
            ica.save(fname)

        # Apply ICA exclusion
        reconst_raw = mne_array.copy()
        exclude_raw = mne_array.copy()
        print("ICAs excluded: ", ica.exclude)
        ica.apply(reconst_raw)

        if not skip_plots:
            # change exclude to all except chosen ICs
            all_ICs = list(range(ica.n_components_))
            for i in ica.exclude:
                all_ICs.remove(i)
            ica.exclude = all_ICs
            ica.apply(exclude_raw)

            # Plot excluded ICAs
            scalings = dict(
                mag=1e-12,
                grad=4e-11,
                eeg=20e-6,
                eog=150e-6,
                ecg=5e-4,
                emg=1e-3,
                ref_meg=1e-12,
                misc=1e-3,
                stim=1,
                resp=1,
                chpi=1e-4,
                whitened=1e2,
            )
            max_val = 1.8 * np.max(np.abs(exclude_raw.get_data(stop=100)))
            scalings["eeg"] = max_val

            f1 = exclude_raw.plot(
                block=True,
                show=show,
                clipping="transparent",
                duration=100,
                title="LFP signal excluded from {}".format(signals[0].source_file),
                remove_dc=False,
                scalings=scalings,
            )

            scalings = dict(
                mag=1e-12,
                grad=4e-11,
                eeg=20e-6,
                eog=150e-6,
                ecg=5e-4,
                emg=1e-3,
                ref_meg=1e-12,
                misc=1e-3,
                stim=1,
                resp=1,
                chpi=1e-4,
                whitened=1e2,
            )
            max_val = 1.8 * np.max(np.abs(reconst_raw.get_data(stop=100)))
            scalings["eeg"] = max_val

            # Plot reconstructed signals w/o excluded ICAs
            f2 = reconst_raw.plot(
                block=True,
                show=show,
                clipping="transparent",
                duration=100,
                title="Reconstructed LFP Data from {}".format(signals[0].source_file),
                remove_dc=False,
                scalings=scalings,
            )

            figs = [f1, f2]
        else:
            figs = [None, None]

        output_dict = OrderedDict()

        signals_grouped_by_region = signals.split_into_groups("region")
        for region, (signals, idxs) in signals_grouped_by_region.items():
            if chans_to_pick is not None:
                idxs_to_use = [x for x in idxs if x in chans_to_pick]
            else:
                idxs_to_use = idxs
            data_to_use = reconst_raw.get_data()[idxs_to_use]
            val, _ = average_signals(data_to_use, clean=False)
            eeg = simuran.Eeg()
            eeg.from_numpy(val * u.V, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            output_dict[region] = eeg

        return reconst_raw, output_dict, figs


class LFPZscoreCombiner(LFPCombiner):
    def combine(self):
        z_threshold = method_kwargs.get("z_threshold", 1.1)
        result, bad_chans, zscores = self.z_score_method(
            signals,
            min_f,
            max_f,
            clean=True,
            z_threshold=z_threshold,
            **filter_kwargs,
        )
        results["zscored"] = zscores

    def z_score_method(
        self, signals, min_f, max_f, clean=True, z_threshold=1.1, **filter_kwargs
    ):
        lfp_signals = signals

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()
        z_score_dict = OrderedDict()

        bad_chans = []
        for region, (signals, _) in signals_grouped_by_region.items():
            val, bad_idx, zscores = z_score_signals(
                [s.samples for s in signals],
                z_threshold=z_threshold,
                verbose=True,
                clean=clean,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            if min_f is not None:
                eeg.filter(min_f, max_f, inplace=True, **filter_kwargs)
            output_dict[region] = eeg
            z_score_dict[region] = zscores
            bad_chans += [signals[i].channel for i in bad_idx]

        return output_dict, bad_chans, z_score_dict


class LFPPickCombiner(LFPCombiner):
    """TODO either integrate pick into zscore + average or make two subclasses"""

    def combine(self):
        if self.method == "pick":
            channels = method_kwargs.get("channels")
            prop = method_kwargs.get("pick_property", "channel")
            if channels is None:
                raise ValueError("You must pass the keyword arg channels for pick")
            container = simuran.GenericContainer(signals[0].__class__)
            container.container = [s for s in signals if getattr(s, prop) in channels]
            result, extra_bad = self.avg_method(
                container, min_f, max_f, clean=True, **filter_kwargs
            )
            bad_chans = [s.channel for s in signals if getattr(s, prop) not in channels]
            bad_chans += [signals[i].channel for i in extra_bad]
            results["bad_channels"] = bad_chans
        elif self.method == "pick_zscore":
            channels = method_kwargs.get("channels")
            prop = method_kwargs.get("pick_property", "channel")
            if channels is None:
                raise ValueError("You must pass the keyword arg channels for pick")
            container = simuran.GenericContainer(signals[0].__class__)
            container.container = [s for s in signals if getattr(s, prop) in channels]
            result, extra_bad, _ = self.z_score_method(
                container, min_f, max_f, clean=True, **filter_kwargs
            )
            if len(extra_bad) != 0:
                if isinstance(data, simuran.Recording):
                    msg = f"Signals from {data.source_file} -- {extra_bad} don't agree"
                else:
                    msg = f"Signals {extra_bad} don't agree"
                print(msg)
            bad_chans = [s.channel for s in signals if getattr(s, prop) not in channels]
            bad_chans += [signals[i].channel for i in extra_bad]
            results["bad_channels"] = bad_chans


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

    """
    avg_sig = np.mean(signals, axis=0)
    std_sig = np.std(signals, axis=0)
    # Use this with axis = 0 for per signal
    std_sig = np.where(std_sig == 0, 1, std_sig)

    z_scores = np.zeros(shape=(len(signals), len(signals[0])))

    for i, s in enumerate(signals):
        z_scores[i] = (s - avg_sig) / std_sig

    z_score_abs = np.abs(z_scores)

    z_score_means = np.nanmean(z_score_abs, axis=1)
    z_threshold = z_threshold * np.median(z_score_means)

    good, bad = [], []
    for i, val in enumerate(z_score_means):
        if val > z_threshold:
            bad.append(i)
        else:
            good.append(i)

    good_signals = np.array([signals[i] for i in good])
    bad_signals = np.array([signals[i] for i in bad])

    if len(good) == 0:
        raise RuntimeError(f"No good signals found, bad were {bad}")

    return good_signals, bad_signals, good, bad, z_scores


def average_signals(signals, z_threshold=1.1, verbose=False, clean=True):
    """
    Clean and average a set of signals.

    Parameters
    ----------
    signals : iterable
        Assumed to be an N_chans * N_samples iterable.
    sampling_rate : int
        The sampling rate of the signals in samples/s.
    filter_ : tuple
        Butter filter parameters.
    z_threshold : float, optional.
        The threshold for the mean signal z-score to be an outlier.
        Defaults to 1.1. This means z > 1.1 * z.median is outlier.
    verbose : bool, optional.
        Whether to print further information, defaults to False.

    Returns
    -------
    np.ndarray
        The cleaned and averaged signals.

    """
    if type(signals) is not np.ndarray:
        signals_ = np.array(signals)
    else:
        signals_ = signals

    # 1. Try to identify dead channels
    if clean:
        good_signals, bad_signals, good_idx, bad_idx, _ = detect_outlying_signals(
            signals_, z_threshold=z_threshold
        )
        if verbose:
            if len(bad_idx) != 0:
                print(
                    "Excluded {} signals with indices {}".format(len(bad_idx), bad_idx)
                )
    else:
        good_signals = signals
        bad_idx = []

    # 1a. Consider trying to remove noise per channel? Or after avg?

    # 2. Average the good signals
    avg_sig = np.mean(good_signals, axis=0)

    if hasattr(signals[0], "unit"):
        return (avg_sig * signals[0].unit), bad_idx
    else:
        return avg_sig, bad_idx


def z_score_signals(signals, z_threshold=1.1, verbose=False, clean=True):
    if type(signals) is not np.ndarray:
        signals_ = np.array(signals)
    else:
        signals_ = signals

    # Like this will z-score before check
    # for i in range(len(signals_)):
    #     div = np.std(signals_[i])
    #     div = np.where(div == 0, 1, div)
    #     signals_[i] = (signals_[i] - np.mean(signals_[i])) / div

    # 1. Try to identify dead channels
    if clean:
        good_signals, bad_signals, good_idx, bad_idx, _ = detect_outlying_signals(
            signals_, z_threshold=z_threshold
        )
        if verbose:
            if len(bad_idx) != 0:
                print(
                    "Excluded {} signals with indices {}".format(len(bad_idx), bad_idx)
                )
    else:
        bad_idx = []

    for i in range(len(signals_)):
        if i not in bad_idx:
            div = np.std(signals_[i])
            div = np.where(div == 0, 1, div)
            signals_[i] = (signals_[i] - np.mean(signals_[i])) / div
    res = np.mean(signals_, axis=0)

    # Technically, the signals are now dimensionless
    # Including unit for compatibability
    if hasattr(signals[0], "unit"):
        return (res * signals[0].unit), bad_idx, signals_
    else:
        return res, bad_idx, signals_


class LFPClean(object):
    """
    Class to clean LFP signals.

    Attributes
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg", "zscore", "avg_raw", "ica", "pick".
    visualise : bool
        Whether to visualise the cleaning.
    show_vis : bool
        Whether to visualise on the fly or return figs

    Parameters
    ----------
    method : string
        The method to use for cleaning.
        Currently supports "avg", "zscore", "avg_raw", "ica", "pick".
    visualise : bool
        Whether to visualise the cleaning.
    show_vis : bool
        Whether to visualise on the fly or return figs

    Methods
    -------
    clean(recording/signals)

    """

    def __init__(self, method="avg", visualise=False, show_vis=True):
        self.method = method
        self.visualise = visualise
        self.show_vis = show_vis

    def compare_methods(self, methods, data, min_f, max_f, **filter_kwargs):
        results = {}
        temp = self.visualise
        for method in methods:
            self.method = method
            self.visualise = False
            result = self.clean(data, min_f, max_f, **filter_kwargs)["signals"]
            for k, v in result.items():
                v.channel = method[:5]
                results[f"{method}-{k}"] = v
        self.visualise = temp

        if isinstance(data, simuran.Recording):
            signals = data.data["signals"]
        else:
            signals = data

        fig = self.vis_cleaning(results, signals)

        fig = simuran.SimuranFigure(fig, done=True)

        return fig

    def clean(self, data, min_f=None, max_f=None, method_kwargs=None, **filter_kwargs):
        """
        Clean the lfp signals.

        Parameters
        ----------
        data : simuran.recording.Recording or simuran.EegArray
            also accepts simuran.GenericContainer of simuran.BaseSignal
            The signals to clean

        Returns
        -------
        dict
            keys are "signals", "fig", "cleaned", "zscored"

        """
        bad_chans = None
        results = {
            "signals": {},
            "fig": None,
            "cleaned": None,
            "zscored": None,
            "bad_channels": bad_chans,
        }
        if method_kwargs is None:
            method_kwargs = {}
        ica_fname = None
        if isinstance(data, simuran.Recording):
            signals = data.data["signals"]
            base_dir = method_kwargs.get("base_dir", None)
            ica_fname = data.get_name_for_save(base_dir)
            for i in range(len(signals)):
                signals[i].load()
        else:
            signals = data

        highpass = 0.0
        if min_f is not None:
            filter_kwargs["verbose"] = filter_kwargs.get("verbose", "WARNING")
            signals = self.filter_sigs(signals, min_f, max_f, **filter_kwargs)
            highpass = min_f

        if self.method == "avg":
            z_threshold = method_kwargs.get("z_threshold", 1.1)
            result, bad_chans = self.avg_method(
                signals,
                min_f,
                max_f,
                clean=True,
                z_threshold=z_threshold,
                **filter_kwargs,
            )
        elif self.method == "zscore":
            z_threshold = method_kwargs.get("z_threshold", 1.1)
            result, bad_chans, zscores = self.z_score_method(
                signals,
                min_f,
                max_f,
                clean=True,
                z_threshold=z_threshold,
                **filter_kwargs,
            )
            results["zscored"] = zscores
        elif self.method == "avg_raw":
            result, _ = self.avg_method(
                signals, min_f, max_f, clean=False, **filter_kwargs
            )
        elif self.method == "ica":
            channels = method_kwargs.get("channels")
            prop = method_kwargs.get("pick_property", "channel")
            manual_ica = method_kwargs.get("manual_ica", False)
            if channels is not None:
                idxs = [
                    i
                    for i in range(len(signals))
                    if getattr(signals[i], prop) in channels
                ]
                bad_chans = [
                    s.channel for s in signals if getattr(s, prop) not in channels
                ]
                results["bad_channels"] = bad_chans
            else:
                idxs = None
            reconst, result, figs = self.ica_method(
                signals,
                chans_to_pick=idxs,
                ica_fname=ica_fname,
                manual=manual_ica,
                highpass=highpass,
            )
            results["cleaned"] = reconst
            results["ica_figs"] = figs
        elif self.method == "pick":
            channels = method_kwargs.get("channels")
            prop = method_kwargs.get("pick_property", "channel")
            if channels is None:
                raise ValueError("You must pass the keyword arg channels for pick")
            container = simuran.GenericContainer(signals[0].__class__)
            container.container = [s for s in signals if getattr(s, prop) in channels]
            result, extra_bad = self.avg_method(
                container, min_f, max_f, clean=True, **filter_kwargs
            )
            bad_chans = [s.channel for s in signals if getattr(s, prop) not in channels]
            bad_chans += [signals[i].channel for i in extra_bad]
            results["bad_channels"] = bad_chans
        elif self.method == "pick_zscore":
            channels = method_kwargs.get("channels")
            prop = method_kwargs.get("pick_property", "channel")
            if channels is None:
                raise ValueError("You must pass the keyword arg channels for pick")
            container = simuran.GenericContainer(signals[0].__class__)
            container.container = [s for s in signals if getattr(s, prop) in channels]
            result, extra_bad, _ = self.z_score_method(
                container, min_f, max_f, clean=True, **filter_kwargs
            )
            if len(extra_bad) != 0:
                if isinstance(data, simuran.Recording):
                    msg = f"Signals from {data.source_file} -- {extra_bad} don't agree"
                else:
                    msg = f"Signals {extra_bad} don't agree"
                print(msg)
            bad_chans = [s.channel for s in signals if getattr(s, prop) not in channels]
            bad_chans += [signals[i].channel for i in extra_bad]
            results["bad_channels"] = bad_chans
        else:
            raise ValueError(f"{self.method} is not a valid clean method")

        results["signals"] = result
        if self.visualise:
            kwargs = {}
            fig = self.vis_cleaning(result, signals, bad_chans=bad_chans, **kwargs)

            fig = simuran.SimuranFigure(fig, done=True)

            results["fig"] = fig

        return results

    def vis_cleaning(self, result, signals, bad_chans=None, **kwargs):
        if self.method == "zscore":
            signals_ = deepcopy(signals)
            for i in range(len(signals_)):
                val = np.array(signals_[i].samples)
                div = np.std(val)
                div = np.where(div == 0, 1, div)
                mean = np.mean(val)
                signals_[i].samples = ((val - mean) / div) * signals_[i].samples.unit
        else:
            signals_ = signals

        if isinstance(result, dict):
            eeg_array = simuran.EegArray()
            _, eeg_idxs = signals_.group_by_property("channel_type", "eeg")
            eeg_sigs = signals_.subsample(idx_list=eeg_idxs, inplace=False)
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in eeg_sigs])

            for k, v in result.items():
                eeg_array.append(v)
        else:
            eeg_array = simuran.EegArray()
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in result])

        fig = eeg_array.plot(proj=False, show=self.show_vis, bad_chans=bad_chans)

        return fig

    def z_score_method(
        self, signals, min_f, max_f, clean=True, z_threshold=1.1, **filter_kwargs
    ):
        lfp_signals = signals

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()
        z_score_dict = OrderedDict()

        bad_chans = []
        for region, (signals, _) in signals_grouped_by_region.items():
            val, bad_idx, zscores = z_score_signals(
                [s.samples for s in signals],
                z_threshold=z_threshold,
                verbose=True,
                clean=clean,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            if min_f is not None:
                eeg.filter(min_f, max_f, inplace=True, **filter_kwargs)
            output_dict[region] = eeg
            z_score_dict[region] = zscores
            bad_chans += [signals[i].channel for i in bad_idx]

        return output_dict, bad_chans, z_score_dict

    def avg_method(
        self, signals, min_f, max_f, clean=True, z_threshold=1.1, **filter_kwargs
    ):
        lfp_signals = signals

        signals_grouped_by_region = lfp_signals.split_into_groups("region")

        output_dict = OrderedDict()

        bad_chans = []
        for region, (signals, _) in signals_grouped_by_region.items():
            val, bad_idx = average_signals(
                [s.samples for s in signals],
                z_threshold=z_threshold,
                verbose=True,
                clean=clean,
            )
            eeg = simuran.Eeg()
            eeg.from_numpy(val, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            if min_f is not None:
                eeg.filter(min_f, max_f, inplace=True, **filter_kwargs)
            output_dict[region] = eeg
            bad_chans += [signals[i].channel for i in bad_idx]

        return output_dict, bad_chans

    def ica_method(
        self,
        signals,
        exclude=None,
        chans_to_pick=None,
        save=True,
        ica_fname=None,
        manual=True,
        highpass=0.0,
    ):
        skip_plots = not self.visualise
        show = self.show_vis
        if not isinstance(signals, simuran.EegArray):
            eeg_array = simuran.EegArray()
            eeg_array.set_container([simuran.Eeg(signal=eeg) for eeg in signals])

        regions = list(set([s.region for s in signals]))

        mne_array = signals.convert_signals_to_mne(verbose=False)
        mne_array.info["highpass"] = highpass

        loaded = False
        if save:
            home = os.path.abspath(os.path.expanduser("~"))
            default_save_location = os.path.join(home, ".skm_python", "ICA_files")
            os.makedirs(default_save_location, exist_ok=True)
            bit = "-auto"
            if manual:
                bit = ""
            if ica_fname is None:
                if os.path.basename(signals[0].source_file) != "<unknown>":
                    ica_fname = (
                        os.path.splitext(os.path.basename(signals[0].source_file))[0]
                        + f"{bit}-ica.fif.gz"
                    )
                else:
                    val = np.round(np.mean(signals[0].samples), 3)
                    name = str(val).replace(".", "-")
                    ica_fname = os.path.join(name) + f"{bit}-ica.fif.gz"
            elif not ica_fname.endswith("-ica.fif.gz"):
                if not ica_fname.endswith("-ica.fif"):
                    ica_fname = os.path.splitext(ica_fname)[0] + f"{bit}-ica.fif.gz"

            fname = os.path.join(default_save_location, ica_fname)

            if os.path.exists(fname):
                print("Loading ICA from {}".format(fname))
                ica = read_ica(fname, verbose="ERROR")
                loaded = True
                print(ica.exclude)

        if not loaded:
            ica = ICA(
                method="picard",
                random_state=42,
                max_iter="auto",
                fit_params=dict(ortho=True, extended=True),
                n_components=None,
            )

            ica.fit(mne_array)
            if manual:

                if exclude is None:
                    # Plot raw ICAs
                    ica.plot_sources(mne_array)

                    cont = input("Plot region overlay? (y|n) \n")
                    if cont.strip().lower() == "y":
                        reg_grps = []
                        for reg in regions:
                            temp_grp = []
                            for ch in mne_array.info.ch_names:
                                if reg in ch:
                                    temp_grp.append(ch)
                            reg_grps.append(temp_grp)
                        for grps in reg_grps:
                            ica.plot_overlay(
                                mne_array,
                                stop=int(30 * 250),
                                title="{}".format(grps[0][:3]),
                                picks=grps,
                            )
                else:
                    # ICAs to exclude
                    ica.exclude = exclude
                    if not skip_plots:
                        ica.plot_sources(mne_array)

            else:
                end_time = min(120.0, signals[0].get_end().value)
                ica = ica.detect_artifacts(
                    mne_array, start_find=20.0, stop_find=end_time
                )

        if save:
            ica.save(fname)

        # Apply ICA exclusion
        reconst_raw = mne_array.copy()
        exclude_raw = mne_array.copy()
        print("ICAs excluded: ", ica.exclude)
        ica.apply(reconst_raw)

        if not skip_plots:
            # change exclude to all except chosen ICs
            all_ICs = list(range(ica.n_components_))
            for i in ica.exclude:
                all_ICs.remove(i)
            ica.exclude = all_ICs
            ica.apply(exclude_raw)

            # Plot excluded ICAs
            scalings = dict(
                mag=1e-12,
                grad=4e-11,
                eeg=20e-6,
                eog=150e-6,
                ecg=5e-4,
                emg=1e-3,
                ref_meg=1e-12,
                misc=1e-3,
                stim=1,
                resp=1,
                chpi=1e-4,
                whitened=1e2,
            )
            max_val = 1.8 * np.max(np.abs(exclude_raw.get_data(stop=100)))
            scalings["eeg"] = max_val

            f1 = exclude_raw.plot(
                block=True,
                show=show,
                clipping="transparent",
                duration=100,
                title="LFP signal excluded from {}".format(signals[0].source_file),
                remove_dc=False,
                scalings=scalings,
            )

            scalings = dict(
                mag=1e-12,
                grad=4e-11,
                eeg=20e-6,
                eog=150e-6,
                ecg=5e-4,
                emg=1e-3,
                ref_meg=1e-12,
                misc=1e-3,
                stim=1,
                resp=1,
                chpi=1e-4,
                whitened=1e2,
            )
            max_val = 1.8 * np.max(np.abs(reconst_raw.get_data(stop=100)))
            scalings["eeg"] = max_val

            # Plot reconstructed signals w/o excluded ICAs
            f2 = reconst_raw.plot(
                block=True,
                show=show,
                clipping="transparent",
                duration=100,
                title="Reconstructed LFP Data from {}".format(signals[0].source_file),
                remove_dc=False,
                scalings=scalings,
            )

            figs = [f1, f2]
        else:
            figs = [None, None]

        output_dict = OrderedDict()

        signals_grouped_by_region = signals.split_into_groups("region")
        for region, (signals, idxs) in signals_grouped_by_region.items():
            if chans_to_pick is not None:
                idxs_to_use = [x for x in idxs if x in chans_to_pick]
            else:
                idxs_to_use = idxs
            data_to_use = reconst_raw.get_data()[idxs_to_use]
            val, _ = average_signals(data_to_use, clean=False)
            eeg = simuran.Eeg()
            eeg.from_numpy(val * u.V, sampling_rate=signals[0].sampling_rate)
            eeg.region = region
            eeg.channel = "avg"
            output_dict[region] = eeg

        return reconst_raw, output_dict, figs

    def filter_sigs(self, signals, min_f, max_f, **filter_kwargs):
        eeg_array = simuran.EegArray()
        for signal in signals:
            filt_s = signal.filter(min_f, max_f, inplace=False, **filter_kwargs)
            eeg_array.append(simuran.Eeg(signal=filt_s))
        return eeg_array
