import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran
from fooof import FOOOFGroup
from simuran.bridges.neurochat_bridge import signal_to_neurochat

from frequency_analysis import calculate_psd
from lfp_clean import LFPAverageCombiner, NCSignalSeries

os.chdir(r"E:\Repos\atn-sub-lfp-workflow")


def setup_signals():
    """Set up the signals (such as eeg or lfp)."""

    # The total number of signals in the recording
    num_signals = 32

    # What brain region each signal was recorded from
    regions = ["CA1"] * 32

    # If the wires were bundled, or any other kind of grouping existed
    # If no grouping, grouping = [i for in range(num_signals)]
    groups = ["LFP", "LFP", "LFP", "LFP"] + list(range(num_signals - 4))

    # The sampling rate in Hz of each signal
    sampling_rate = [250] * num_signals

    # This just passes the information on
    output_dict = {
        "num_signals": num_signals,
        "region": regions,
        "group": groups,
        "sampling_rate": sampling_rate,
    }

    return output_dict


def setup_units():
    """Set up the single unit data."""
    # The number of tetrodes, probes, etc - any kind of grouping
    num_groups = 8

    # The region that each group belongs to
    regions = ["CA1"] * num_groups

    # A group number for each group, for example the tetrode number
    groups = [1, 2, 3, 4, 9, 10, 11, 12]

    output_dict = {
        "num_groups": num_groups,
        "region": regions,
        "group": groups,
    }

    return output_dict


def setup_spatial():
    """Set up the spatial data."""

    output_dict = {
        "arena_size": "default",
    }
    return output_dict


def powers(recording, fmin=0.5, fmax=120, **kwargs):
    ss = NCSignalSeries(recording)
    ss.filter(fmin, fmax)
    lc = LFPAverageCombiner(remove_outliers=True)
    signals_grouped_by_region = lc.combine(ss)
    theta_min = kwargs.get("theta_min", 6)
    theta_max = kwargs.get("theta_max", 10)
    delta_min = kwargs.get("delta_min", 1.5)
    delta_max = kwargs.get("delta_max", 4.0)
    low_gamma_min = kwargs.get("low_gamma_min", 30)
    low_gamma_max = kwargs.get("low_gamma_max", 55)
    high_gamma_min = kwargs.get("high_gamma_min", 65)
    high_gamma_max = kwargs.get("high_gamma_max", 90)

    results = {}
    window_sec = 2
    simuran.set_plot_style()

    for name, signal in signals_grouped_by_region.items():
        avg_sig = signal["average_signal"]
        avg_sig = simuran.Eeg.from_numpy(avg_sig, ss.sampling_rate)
        sig_in_use = signal_to_neurochat(avg_sig)
        delta_power = sig_in_use.bandpower(
            [delta_min, delta_max], window_sec=window_sec, band_total=True
        )
        theta_power = sig_in_use.bandpower(
            [theta_min, theta_max], window_sec=window_sec, band_total=True
        )
        low_gamma_power = sig_in_use.bandpower(
            [low_gamma_min, low_gamma_max], window_sec=window_sec, band_total=True
        )
        high_gamma_power = sig_in_use.bandpower(
            [high_gamma_min, high_gamma_max], window_sec=window_sec, band_total=True
        )

        if not (
            delta_power["total_power"]
            == theta_power["total_power"]
            == low_gamma_power["total_power"]
            == high_gamma_power["total_power"]
        ):
            raise ValueError("Unequal total powers")

        results[f"{name} delta"] = delta_power["bandpower"]
        results[f"{name} theta"] = theta_power["bandpower"]
        results[f"{name} low gamma"] = low_gamma_power["bandpower"]
        results[f"{name} high gamma"] = high_gamma_power["bandpower"]
        results[f"{name} total"] = delta_power["total_power"]

        results[f"{name} delta rel"] = delta_power["relative_power"]
        results[f"{name} theta rel"] = theta_power["relative_power"]
        results[f"{name} low gamma rel"] = low_gamma_power["relative_power"]
        results[f"{name} high gamma rel"] = high_gamma_power["relative_power"]

        welch_info = calculate_psd(
            sig_in_use.get_samples(),
            sig_in_use.get_sampling_rate(),
            fmin,
            fmax,
            scale="decibels",
        )

        results[f"{name} welch f"] = welch_info[0]
        results[f"{name} welch p"] = welch_info[1]
        results[f"{name} welch max"] = welch_info[2]
    return results


def main(input_dir, out_dir, cfg_path, overwrite=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = simuran.config_from_file(cfg_path)

    if overwrite or not (out_dir / "results.csv").is_file():
        mapping = {
            "signals": setup_signals(),
            "units": setup_units(),
            "spatial": setup_spatial(),
        }
        ph = simuran.ParamHandler(mapping, name="mapping")
        ph.write(out_dir / "layout.yml")
        rc = load_rc(input_dir, out_dir)
        df = run_analysis(rc, input_dir, config)
        df.to_csv(out_dir / "results.csv", index=True)
    visualise(out_dir, config)


def run_analysis(rc, input_dir, config):
    results = {
        r.get_name_for_save(input_dir): powers(r, **config) for r in rc.load_iter()
    }

    return pd.DataFrame(results).T


def load_rc(path_dir, out_dir):
    from simuran.loaders.neurochat_loader import NCLoader

    loader = NCLoader(system="Axona", pos_extension=".pos")
    files_df = loader.index_files(path_dir)
    files_df["mapping"] = out_dir / "layout.yml"
    return simuran.RecordingContainer.from_table(files_df, loader)


class UnicodeGrabber(object):
    """This is a fully static class to get unicode chars for plotting."""

    char_dict = {
        "micro": "\u00B5",
        "pow2": "\u00B2",
    }

    @staticmethod
    def get_chars():
        return list(UnicodeGrabber.char_dict.keys())

    @staticmethod
    def get(char, default=""):
        return UnicodeGrabber.char_dict.get(char, default)


def visualise(out_dir, config):
    df = pd.read_csv(out_dir / "results.csv")
    welch_freqs = []
    welch_powers = []
    max_pxxs = []

    def str_to_float(a):
        ret = []
        a = a[1:-1].split(" ")
        for val in a:
            try:
                converted = float(val)
                ret.append(converted)
            except ValueError:
                continue
        return np.array(ret)

    for index, row in df.iterrows():
        welch_freqs.append(str_to_float(row["CA1 welch f"]))
        welch_powers.append(str_to_float(row["CA1 welch p"]))
        max_pxxs.append(row["CA1 welch max"])
    in_list = [np.array(welch_freqs).flatten(), np.array(welch_powers).flatten()]

    df = pd.DataFrame(in_list).T
    df.columns = ["frequency", "power"]
    df.to_csv(os.path.join(out_dir, "power_results.csv"), index=False)

    ## Then use seaborn to produce a summary plot
    scale = config["psd_scale"]
    simuran.set_plot_style()
    plt.close("all")
    for ci, oname in zip([95, None], ["_ci", ""]):
        out_loc = os.path.join(out_dir, f"ca1_power_final{oname}.pdf")

        sns.lineplot(
            data=df,
            x="frequency",
            y="power",
            ci=ci,
            estimator=np.median,
        )
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, config["max_psd_freq"])
        plt.ylim(-40, 0)
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            plt.ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            plt.ylabel("PSD (dB)")
        else:
            raise ValueError(f"Unsupported scale {scale}")
        plt.title("CA1 LFP power (median)")
        simuran.despine()

        plt.savefig(
            out_loc,
            dpi=400,
        )
        print(f"Figure saved to {out_loc}")
        plt.close("all")

    fg = FOOOFGroup(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=2,
        min_peak_height=0.1,
        peak_threshold=2.0,
        aperiodic_mode="fixed",
    )

    fooof_arr_s = np.array(welch_powers)
    for i in range(len(fooof_arr_s)):
        fooof_arr_s[i] = np.power(10.0, (fooof_arr_s[i] / 10.0)) * max_pxxs[i]

    fooof_arr_f = np.array(welch_freqs)
    fg.fit(fooof_arr_f[0], fooof_arr_s, [0.5, 120], progress="tqdm")

    peaks = fg.get_params("peak_params", 0)[:, 0]

    peaks_data = [[p, "Control", "CA1"] for p in peaks]
    peaks_df = pd.DataFrame.from_records(
        peaks_data, columns=["Peak frequency", "Group", "Region"]
    )

    fig, ax = plt.subplots()
    sns.histplot(
        data=peaks_df,
        x="Peak frequency",
        ax=ax,
    )
    simuran.despine()
    out_name = os.path.join(out_dir, "ca1_peaks_fooof.pdf")
    fig.savefig(out_name, dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    try:
        simuran.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.config["ca1_directory"],
            snakemake.output[0],
            snakemake.config["simuran_config"],
        )
    except Exception:
        input_dir = r"H:\ATN_CA1"
        output_dir = "results\ca1_analysis"
        config_loc = r"config\simuran_params.yml"
        main(input_dir, output_dir, config_loc)
