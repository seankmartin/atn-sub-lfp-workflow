from copy import copy
import logging
import os
from math import floor, ceil
from pathlib import Path
from pprint import pprint
import csv
import argparse
from random import random

import simuran as smr
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from neurochat.nc_lfp import NLfp
import numpy as np
from scipy.signal import coherence
from skm_pyutils.py_table import list_to_df, df_from_file, df_to_file
from skm_pyutils.py_config import parse_args
import seaborn as sns
from scipy.signal import welch

from neuronal.decoding import LFPDecoder

module_logger = logging.getLogger("simuran.custom.tmaze_analyse")


def decoding(lfp_array, groups, labels, base_dir):
    for group in ["Control", "Lesion (ATNx)"]:
        correct_groups = groups == group
        lfp_to_use = lfp_array[correct_groups, :]
        labels_ = labels[correct_groups]

        decoder = LFPDecoder(
            labels=labels_,
            mne_epochs=None,
            features=lfp_to_use,
            cv_params={"n_splits": 100},
        )
        out = decoder.decode()
        print(decoder.decoding_accuracy(out[2], out[1]))

        print("\n----------Cross Validation-------------")
        decoder.cross_val_decode(shuffle=False)
        pprint(decoder.cross_val_result)
        pprint(decoder.confidence_interval_estimate("accuracy"))

        print("\n----------Cross Validation Control (shuffled)-------------")
        decoder.cross_val_decode(shuffle=True)
        pprint(decoder.cross_val_result)
        pprint(decoder.confidence_interval_estimate("accuracy"))

        random_search = decoder.hyper_param_search(verbose=True, set_params=False)
        print("Best params:", random_search.best_params_)

        decoder.visualise_features(output_folder=base_dir, name=f"_{group}")


def main(
    tmaze_times_filepath,
    config_filepath,
    out_dir,
    do_coherence=True,
    do_decoding=True,
    overwrite=False,
):
    config = smr.parse_config(config_filepath)
    datatable = df_from_file(tmaze_times_filepath)
    rc = smr.RecordingContainer.from_table(datatable, smr.loader("nwb"))

    if not should_skip(out_dir, overwrite):
        compute_and_save_coherence(out_dir, do_coherence, config, rc)
    coh_df, power_df, res_df, groups, choices, new_lfp = load_saved_results(
        out_dir, config
    )
    
    if do_coherence:
        plot_coherence_results(res_df, coh_df, power_df, out_dir)
    if do_decoding:
        groups = np.array(groups)
        labels = np.array(choices)
        decoding(new_lfp, groups, labels, out_dir)

def compute_and_save_coherence(out_dir, config, rc):
    results, coherence_df_list = [], []
    for r in rc.load_iter():
        module_logger.info(f"Analysing t_maze for {r.source_file}")
        result, coherence = compute_coherence_and_power(r, config, out_dir)
        results.append(result)
        coherence_df_list.extend(coherence)
    save_computed_info()


def should_skip(out_dir: "Path", overwrite):
    decoding_loc = out_dir / "decoding.csv"
    coherence_loc = out_dir / "coherence.csv"
    return decoding_loc.is_file() and (not overwrite) and coherence_loc.is_file()


def compute_coherence_and_power(r, config, out_dir):
    sub_lfp, rsc_lfp, fs, duration = extract_lfp_info(r)
    sig_dict = {"SUB": sub_lfp, "RSC": rsc_lfp}

    compute_per_trial_coherence_power(r, config, fs, duration, sig_dict, out_dir)

    if do_decoding:
        groups.append(group)
        choices.append(str(r.passed).strip())


def compute_per_trial_coherence_power(r, config, fs, duration, sig_dict, out_dir):
    coherence_res = calculate_coherence(sig_dict["SUB"], sig_dict["RSC"], fs, config)
    results_list, coherence_list, pxx_list = [], [], []

    for i, trial_type in zip(range(1, 3), ("forced", "choice")):
        lfp_portions, final_trial_type, group = setup_recording_info(
            r, fs, duration, i, trial_type, config
        )
        for k, bounds in lfp_portions.items():
            res_list = copy(coherence_res)
            fn_params = [r, config, fs, sig_dict, final_trial_type, group, k, bounds]
            compute_power_per_trial(*fn_params, pxx_list, res_list)
            sub_lfp, rsc_lfp, f, Cxy = compute_coherence_per_trial(
                *fn_params, coherence_list, res_list
            )
            results_list.append(res_list)

        plot_results_intermittent(r, sub_lfp, rsc_lfp, f, Cxy, out_dir)
    return results_list, coherence_list, pxx_list


def compute_coherence_per_trial(
    r,
    config,
    fs,
    sig_dict,
    final_trial_type,
    group,
    k,
    bounds,
    coherence_list,
    res_list,
):
    sub_lfp, rsc_lfp, f, Cxy = coherence_from_bounds(config, fs, sig_dict, bounds)
    coherence_list.extend(
        make_coherence_tuple(r, final_trial_type, group, k, f_, cxy_)
        for f_, cxy_ in zip(f, Cxy)
    )
    extract_decoding_vals(config, k, f, Cxy)
    res_list.extend(theta_delta(f, Cxy, config))
    res_list.append(group)
    return sub_lfp, rsc_lfp, f, Cxy


def compute_power_per_trial(
    r,
    config,
    fs,
    sig_dict,
    final_trial_type,
    group,
    k,
    bounds,
    pxx_list,
    res_list,
):
    res_dict = {}
    for region, signal in sig_dict.items():
        lfp = convert_signal_to_nc(bounds, signal, fs)
        bandpowers(config, res_dict, k, region, lfp)
    res_list.extend(list_results(r, res_dict, final_trial_type, k))
    f_welch, Pxx = compute_power(fs, sig_dict["SUB"], config)
    pxx_list.extend(
        make_power_tuple(r, final_trial_type, group, k, p_val, f_val)
        for p_val, f_val in zip(Pxx, f_welch)
    )


def make_power_tuple(r, final_trial_type, group, k, p_val, f_val):
    return [
        f_val,
        p_val,
        r.passed.strip(),
        group,
        k,
        final_trial_type,
    ]


def make_coherence_tuple(r, final_trial_type, group, k, f_, cxy_):
    return (
        f_,
        cxy_,
        r.attrs["passed"],
        group,
        r.attrs["trial"],
        r.attrs["session"],
        k,
        final_trial_type,
    )


def setup_recording_info(r, fs, duration, i, trial_type, config):
    max_lfp_lengths_seconds = config.get("max_lfp_lengths")
    time_dict = extract_times_for_lfp(r, fs, duration, i)
    lfp_portions = extract_lfp_portions(
        max_lfp_lengths_seconds, fs, duration, time_dict
    )
    final_trial_type = convert_trial_type(r, trial_type)
    group = get_group(r)
    return lfp_portions, final_trial_type, group


def compute_power(fs, x, config):
    f_welch, Pxx = welch(
        x,
        fs=fs,
        nperseg=config["tmaze_winsec"] * 250,
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    f_welch = f_welch[
        np.nonzero(
            (f_welch >= config["tmaze_minf"]) & (f_welch <= config["tmaze_maxf"])
        )
    ]
    Pxx = Pxx[
        np.nonzero(
            (f_welch >= config["tmaze_minf"]) & (f_welch <= config["tmaze_maxf"])
        )
    ]

    Pxx_max = np.max(Pxx)
    Pxx = 10 * np.log10(Pxx / Pxx_max)
    return f_welch, Pxx


def plot_results_intermittent(r, x, y, f, Cxy, out_dir):
    every_few_iters = random()
    if every_few_iters < 0.05:
        fig2, ax2 = plt.subplots(3, 1)
        ax2[0].plot(f, Cxy, c="k")
        ax2[1].plot([i / 250 for i in range(len(x))], x, c="k")
        ax2[2].plot([i / 250 for i in range(len(y))], y, c="k")
        fig2.savefig(
            out_dir
            / f'coherence_{r.attrs["rat"]}_{r.attrs["session"]}_{r.attrs["trial"]}.png'
        )

        plt.close(fig2)


def get_group(r):
    return "Control" if r.attrs["animal"].lower().startswith("c") else "Lesion (ATNx)"


def save_computed_info(results_list, coherence_df_list, pxx_list, out_dir):
    headers = get_result_headers()
    res_df = pd.DataFrame(results_list, columns=headers)
    df_to_file(res_df, out_name, index=False)

    headers = get_coherence_headers()
    coherence_df = list_to_df(coherence_df_list, headers=headers)
    df_to_file(coherence_df, oname_coherence, index=False)

    headers = get_power_headers()
    power_df = list_to_df(pxx_list, headers)
    df_to_file(power_df, oname_power_tmaze, index=False)
    smr.set_plot_style()


def plot_coherence_results(res_df, coherence_df, power_df, out_dir):
    plot_banded_coherence(out_dir, res_df)
    plot_grouped_power_coherence(out_dir, coherence_df, power_df)
    plot_choice_power(power_df, out_dir)
    plot_coherence_choice(coherence_df, out_dir)


def save_decoding_values(out_dir, overwrite, groups, choices, new_lfp):
    decoding_loc = out_dir / "decoding.csv"
    if not os.path.exists(decoding_loc) or overwrite:
        with open(decoding_loc, "w") as f:
            for i in range(len(groups)):
                line = ""
                line += f"{groups[i]},"
                line += f"{choices[i]},"
                for v in new_lfp[i]:
                    line += f"{v},"
                line = line[:-1] + "\n"
                f.write(line)


def plot_coherence_choice(coherence_df):
    coherence_df["Trial result"] = coherence_df["Trial"]
    coherence_df_sub_bit = coherence_df[
        (coherence_df["Part"] == "choice") & (coherence_df["Trial"] != "Forced")
    ]

    sns.lineplot(
        data=coherence_df_sub_bit,
        x="Frequency (Hz)",
        y="Coherence",
        hue="Group",
        style="Trial result",
        ci=95,
        estimator=np.median,
    )
    plt.ylim(0, 1)
    smr.despine()
    plt.savefig(
        os.path.join(here, "..", "sim_results", "tmaze", "choice_coherence_ci.pdf"),
        dpi=400,
    )
    plt.close("all")


def plot_choice_power(power_df):
    power_df["Trial result"] = power_df["Trial"]
    power_df_sub_bit = power_df[
        (power_df["Part"] == "choice") & (power_df["Trial"] != "Forced")
    ]
    sns.lineplot(
        data=power_df_sub_bit,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Group",
        style="Trial result",
        estimator=np.median,
        ci=95,
    )
    smr.despine()
    plt.savefig(
        os.path.join(here, "..", "sim_results", "tmaze", "choice_power_ci.pdf"),
        dpi=400,
    )
    plt.close("all")


def plot_banded_coherence(out_dir, res_df):
    res_df = res_df[res_df["part"] == "choice"]
    plot_bar_coherence(res_df, "Theta", out_dir)
    plot_bar_coherence(res_df, "Delta", out_dir)


def plot_grouped_power_coherence(out_dir, coherence_df, power_df):
    for group in ("Control", "Lesion (ATNx)"):
        coherence_df_sub = coherence_df[coherence_df["Group"] == group]
        power_df_sub = power_df[power_df["Group"] == group]
        plot_group_coherence(group, coherence_df_sub, out_dir)
        plot_group_power(group, power_df_sub, out_dir)


def plot_group_power(group, power_df_sub, out_dir):
    fig, ax = plt.subplots()
    sns.lineplot(
        data=power_df_sub,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Part",
        style="Trial",
        ci=95,
        estimator=np.median,
        ax=ax,
    )
    ax.set_xlim(0, 40)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_power_ci")
    fig.save()


def plot_group_coherence(group, coherence_df_sub, out_dir):
    fig, ax = plt.subplots()
    for ci, ci_name in zip((None, 95), ("", "_ci")):
        sns.lineplot(
            data=coherence_df_sub,
            x="Frequency (Hz)",
            y="Coherence",
            hue="Part",
            style="Trial",
            ci=ci,
            estimator=np.median,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        smr.despine()
        fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_coherence{ci_name}")
        fig.save()


def plot_bar_coherence(res_df, band: str, out_dir):
    fig, ax = plt.subplots()
    sns.barplot(
        data=res_df,
        x="trial",
        y=f"{band}_coherence",
        hue="Group",
        estimator=np.median,
        ax=ax,
    )
    ax.set_xlabel("Trial result")
    ax.set_ylabel(f"{band} coherence")
    plt.tight_layout()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{band} coherence")
    fig.save()


def get_power_headers():
    return ["Frequency (Hz)", "Power (dB)", "Passed", "Group", "Part", "Trial"]


def get_coherence_headers():
    return [
        "Frequency (Hz)",
        "Coherence",
        "Passed",
        "Group",
        "Test",
        "Session",
        "Part",
        "Trial",
    ]


def get_result_headers():
    # TODO fix the order of these
    return [
        "location",
        "session",
        "animal",
        "test",
        "choice",
        "part",
        "trial",
        "SUB_delta",
        "SUB_theta",
        "RSC_delta",
        "RSC_theta",
        "Theta_coherence",
        "Delta_coherence",
        "Full_theta_coherence",
        "Full_delta_coherence",
        "Peak 12Hz Theta coherence",
        "Group",
    ]


def list_results(r, res_dict, final_trial_type, k):
    res_list = [
        r.location,
        r.session,
        r.animal,
        r.test,
        r.passed,
        k,
        final_trial_type,
    ]
    res_list += [
        res_dict[f"SUB-{k}_delta"],
        res_dict[f"SUB-{k}_theta"],
        res_dict[f"RSC-{k}_delta"],
        res_dict[f"RSC-{k}_theta"],
    ]

    return res_list


def convert_trial_type(r, trial_type):
    if trial_type == "forced":
        return "Forced"
    elif r.passed.strip().upper() == "Y":
        return "Correct"
    elif r.passed.strip().upper() == "N":
        return "Incorrect"
    else:
        return "ERROR IN ANALYSIS"


def theta_delta(f, Cxy, config):
    theta_co = Cxy[np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))]
    delta_co = Cxy[np.nonzero((f >= config["delta_min"]) & (f <= config["delta_max"]))]
    max_theta_coherence = np.nanmean(theta_co)
    max_delta_coherence = np.nanmean(delta_co)

    theta_co_peak = Cxy[np.nonzero((f >= 11.0) & (f <= 13.0))]
    peak_theta_coherence = np.nanmax(theta_co_peak)
    return max_theta_coherence, max_delta_coherence, peak_theta_coherence


def extract_decoding_vals(do_decoding, config, k, f, Cxy):
    if do_decoding:
        if k == "choice":
            coherence_vals_for_decode = Cxy[
                np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))
            ]
            s, e = (k_) * hf, (k_ + 1) * hf
            new_lfp[j, s:e] = coherence_vals_for_decode


def coherence_from_bounds(config, fs, sig_dict, bounds):
    sub_s = sig_dict["SUB"]
    rsc_s = sig_dict["RSC"]
    x = np.array(sub_s[bounds[0] : bounds[1]].to(u.mV))
    y = np.array(rsc_s[bounds[0] : bounds[1]].to(u.mV))

    f, Cxy = coherence(x, y, fs, nperseg=config["tmaze_winsec"] * fs, nfft=256)
    f = f[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    Cxy = Cxy[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    return x, y, f, Cxy


def bandpowers(config, res_dict, k, region, lfp):
    delta_power = lfp.bandpower(
        [config["delta_min"], config["delta_max"]],
        window_sec=config["tmaze_winsec"],
        band_total=True,
    )
    theta_power = lfp.bandpower(
        [config["theta_min"], config["theta_max"]],
        window_sec=config["tmaze_winsec"],
        band_total=True,
    )
    res_dict["{}-{}_delta".format(region, k)] = delta_power["relative_power"]
    res_dict["{}-{}_theta".format(region, k)] = theta_power["relative_power"]


def extract_lfp_portions(max_lfp_lengths_seconds, fs, duration, time_dict):
    lfp_portions = {}
    for k, max_len in max_lfp_lengths_seconds.items():
        start_time, end_time = extract_start_choice_end(
            max_lfp_lengths_seconds, fs, time_dict, k, max_len
        )
        end_time = verify_start_end(fs, duration, start_time, end_time)
        lfp_portions[k] = (start_time, end_time)
    return lfp_portions


def convert_signal_to_nc(bounds, signal, fs):
    lfp_t1, lfp_t2 = bounds
    lfp = NLfp()
    lfp.set_channel_id(signal.channel)
    lfp._timestamp = np.array(signal.timestamps[lfp_t1:lfp_t2].to(u.s))
    lfp._samples = np.array(signal.samples[lfp_t1:lfp_t2].to(u.mV))
    lfp._record_info["Sampling rate"] = fs
    return lfp


def verify_start_end(fs, duration, start_time, end_time):
    """Make sure have at least 1 second and not > duration."""
    if (end_time - start_time) < fs:
        end_time = start_time + fs

    if end_time > int(ceil(duration * 250)):
        raise RuntimeError(
            "End time {} greater than duration {}".format(end_time, duration)
        )

    return end_time


def extract_start_choice_end(max_lfp_lengths_seconds, fs, time_dict, k, max_len):
    """Find start time, choice time, and end time"""
    start_time = time_dict[k][0]
    choice_time = time_dict[k][1]
    end_time = time_dict[k][2]

    if k == "start":
        st = max_lfp_lengths_seconds["choice"][0]
        start_time, end_time = extract_first_times(
            st, fs, max_len, start_time, end_time
        )
    elif k == "choice":
        start_time, end_time = extract_choice_times(
            fs, start_time, choice_time, end_time
        )
    elif k == "end":
        ct = max_lfp_lengths_seconds["choice"][1]
        start_time, end_time = extract_end_times(ct, fs, max_len, start_time, end_time)
    else:
        raise RuntimeError(f"Unsupported key {k}")
    return start_time, choice_time, end_time


def extract_end_times(ct, fs, max_len, start_time, end_time):
    """
    Get end times.

    For the end time, if the end is longer than max_len
    take the first X seconds after the choice data
    """
    start_time = min(start_time + int(ceil(ct * fs)), end_time)
    natural_end_time = start_time + max_len * fs
    end_time = min(natural_end_time, end_time)
    return start_time, end_time


def extract_choice_times(fs, start_time, choice_time, end_time):
    """
    Get choice times.

    For the choice, take (max_len[0], max_len[1]) seconds
    of data around the point.
    """
    left_push = int(floor(v[0] * fs))
    right_push = int(ceil(v[1] * fs))

    start_time = max(choice_time - left_push, start_time)
    end_time = min(choice_time + right_push, end_time)
    return start_time, end_time


def extract_first_times(ct, fs, max_len, start_time, end_time):
    """
    Get start times.

    If the start bit is longer than max_len, take the last X
    seconds before the choice data
    """
    end_time = max(end_time - int(floor(ct * fs)), start_time)
    natural_start_time = end_time - max_len * fs
    start_time = max(natural_start_time, start_time)
    return start_time, end_time


def extract_times_for_lfp(r, fs, duration, i):
    t1 = r.attrs[f"start{i}"]
    t2 = r.attrs[f"choice{i}"]
    t3 = r.attrs[f"end{i}"]

    if t3 > duration:
        raise RuntimeError(f"Last time {t3} greater than duration {duration}")

    lfpt1 = int(floor(t1 * fs))
    lfpt2 = int(ceil(t2 * fs))
    lfpt3 = int(ceil(t3 * fs))

    time_dict = {
        "start": (lfpt1, lfpt2, lfpt2),
        "choice": (lfpt1, lfpt2, lfpt3),
        "end": (lfpt2, lfpt3, lfpt3),
    }
    return time_dict


def extract_lfp_info(r):
    sub_lfp = r.data.processing["average_lfp"]["SUB_avg"].data[:]
    rsc_lfp = r.data.processing["average_lfp"]["RSC_avg"].data[:]
    fs = r.data.processing["average_lfp"]["SUB_avg"].rate
    duration = len(sub_lfp) / fs
    return sub_lfp, rsc_lfp, fs, duration


def calculate_coherence(x, y, fs, config):
    f, Cxy = coherence(x, y, fs, nperseg=config["tmaze_winsec"] * 250)
    f = f[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    Cxy = Cxy[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]

    theta_co = Cxy[np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))]
    delta_co = Cxy[
        np.nonzero((f >= config["delta_min"]) & (f <= config[config["delta_max"]]))
    ]
    theta_coherence = np.nanmean(theta_co)
    delta_coherence = np.nanmean(delta_co)
    return theta_coherence, delta_coherence


def load_saved_results(out_dir,config):
    lfp_len = config["tmaze_lfp_len"]
    decoding_loc = out_dir / "decoding.csv"
    groups, choices, new_lfp = []
    with open(decoding_loc, "r") as f:
        csvreader = csv.reader(f, delimiter=",")
        for row in csvreader:
            groups.append(row[0])
            choices.append(row[1])
            vals = row[2:]
            new_lfp.append(np.array([float(v) for v in vals[:lfp_len]]))

    coh_loc = out_dir / "coherence.csv"
    power_loc = out_dir / "power.csv"
    results_loc = out_dir / "results.csv"
    coherence_df = df_from_file(coh_loc)
    power_df = df_from_file(power_loc)
    res_df = df_from_file(results_loc)

    return coherence_df, power_df, res_df, groups, choices, np.array(new_lfp)


if __name__ == "__main__":
    here_main = os.path.dirname(os.path.abspath(__file__))
    main_output_location = os.path.join(here_main, "results")

    main_xls_location = os.path.join(main_output_location, "tmaze-times.csv")

    parser = argparse.ArgumentParser(description="Tmaze arguments")
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        default="default.py",
        help="path to the configuration file, default.py by default.",
    )
    parser.add_argument(
        "--main_dir",
        "-d",
        type=str,
        default="",
        help="The name of the base directory for the data.",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Whether to overwrite existing output",
    )
    parsed = parse_args(parser, verbose=False)

    cfg_name = parsed.config

    if not os.path.exists(cfg_name):
        cfg_path = os.path.abspath(os.path.join(here_main, "..", "configs", cfg_name))
    else:
        cfg_path = cfg_name
    main_base_dir = parsed.main_dir

    main_plot_individual_sessions = False
    main_do_coherence = True
    main_do_decoding = False

    main_overwrite = parsed.overwrite
    main(
        main_xls_location,
        main_base_dir,
        main_plot_individual_sessions,
        main_do_coherence,
        main_do_decoding,
        main_overwrite,
    )
