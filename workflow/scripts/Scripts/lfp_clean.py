"""Clean LFP signals."""

from abc import ABC, abstractmethod
from collections import OrderedDict

# 1. create object to be worked on - like this
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
import simuran
from mne.filter import filter_data
from mne.preprocessing import ICA, read_ica

from .lfp_utils import (
    average_signals,
    detect_outlying_signals,
    z_score_normalise_signals,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from simuran import Recording


class SignalSeries(ABC):
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
    def __init__(self, recording: "Recording"):
        """Convert recording object into required information."""

    @abstractmethod
    def group_by_brain_region(self):
        """Return signals grouped by the brain region."""

    @abstractmethod
    def data_as_volts(self):
        """Return the data in volts unit (e.g. multiply by 1000)"""

    @property
    def data(self) -> "np.ndarray":
        return self._data

    @data.setter
    def data(self, data: "np.ndarray"):
        self._data = data

    @property
    def description(self) -> "pd.DataFrame":
        return self._description

    @description.setter
    def description(self, description: "pd.DataFrame"):
        self._description = description


class NWBSignalSeries(SignalSeries):
    """LFP is stored in mV units"""

    def __init__(self, recording):
        lfp = recording.data.processing["ecephys"]["LFP"]["ElectricalSeries"]
        self.data = lfp.data[:].T
        self.description = recording.data.electrodes.to_dataframe()
        self.conversion = lfp.conversion
        self.sampling_rate = lfp.rate

    def data_as_volts(self):
        return self.data * self.conversion

    def select_electrodes(self, property_, options):
        """Select electrodes with electrode.property_ in options"""
        to_use = [
            i for i, row in self.description.iterrows() if row[property_] in options
        ]
        self.data = self.data[to_use]
        self.description = self.description.loc[to_use]

    def group_by_brain_region(self, index=False):
        out_dict = {}
        for i, row in self.description.iterrows():
            location = row["location"]
            if location not in out_dict:
                out_dict[location] = []
            to_append = i if index else self.data[i]
            out_dict[location].append(to_append)

        return {k: np.array(v) for k, v in out_dict.items()}

    def filter(self, min_f, max_f, **filter_kwargs):
        """Filters with MNE - kwargs are passed to mne.filter.filter_data"""
        self.data = filter_data(
            self.data_as_volts(), self.sampling_rate, min_f, max_f, **filter_kwargs
        )


class LFPCombiner(ABC):
    """
    Used to combine LFP signals.

    For example, by averaging all LFP signals that fulfil a criteria.
    Or by performing ICA on all the channels for cleaning.
    """

    @abstractmethod
    def combine(self, signals: "SignalSeries"):
        """Combine LFP signals"""

    def vis_combining(self, result, bad_chans=None):
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

        return eeg_array.plot(proj=False, show=self.show_vis, bad_chans=bad_chans)


@dataclass
class LFPAverageCombiner(LFPCombiner):
    """Create "average" signal for each brain region.

    Zscore normalise signals if option is passed.
    Normalisation is performed per brain region.
    """

    z_threshold: float = 1.1
    remove_outliers: bool = False
    z_normalise: bool = True

    def combine(self, signals):
        output_dict = OrderedDict()

        signals_grouped_by_region = signals.group_by_brain_region()
        for region, signals in signals_grouped_by_region.items():
            if self.remove_outliers:
                signals, outliers, good_idx, outliers_idx, _ = detect_outlying_signals(
                    signals, self.z_threshold
                )
            else:
                outliers_idx = np.array([])
                outliers = np.array([])
                good_idx = np.array(list(range(len(signals))))
            signals = (
                z_score_normalise_signals(signals) if self.z_normalise else signals
            )
            average_signal = average_signals(signals)
            output_dict[region] = dict(
                signals=signals,
                average_signal=average_signal,
                outliers=outliers,
                good_idx=good_idx,
                outliers_idx=outliers_idx,
            )

        return output_dict


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
