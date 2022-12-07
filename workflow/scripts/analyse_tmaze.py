import logging
from copy import copy
from math import ceil, floor
from pathlib import Path
from random import random

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr
from neurochat.nc_lfp import NLfp
from scipy.signal import coherence, welch
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

module_logger = logging.getLogger("simuran.custom.tmaze_analyse")


def main(tmaze_times_filepath, config_filepath, out_dir):
    config = smr.config_from_file(config_filepath)
    datatable = df_from_file(tmaze_times_filepath)
    rc = smr.RecordingContainer.from_table(datatable, smr.loader("nwb"))

    compute_and_save_coherence(out_dir, config, rc)


def compute_and_save_coherence(out_dir, config, rc):
    results, coherence_df_list, power_list = [], [], []
    groups, choices = [], []
    new_lfp = np.zeros(shape=(len(rc), 2 * config["tmaze_lfp_len"]), dtype=np.float32)
    for i, r in enumerate(rc.load_iter()):
        module_logger.info(f"Analysing t_maze for {r.source_file}")
        sub_lfp, rsc_lfp, fs, duration = extract_lfp_info(r)
        sig_dict = {"SUB": sub_lfp, "RSC": rsc_lfp}
        result, coherence, power = compute_per_trial_coherence_power(
            r, i, config, fs, duration, sig_dict, out_dir, new_lfp
        )
        if result is None:
            continue
        results.extend(result)
        coherence_df_list.extend(coherence)
        power_list.extend(power)
        groups.append(get_group(r))
        choices.append(r.attrs["passed"])

    save_computed_info(results, coherence_df_list, power_list, out_dir)
    save_decoding_values(out_dir, groups, choices, new_lfp)


def compute_per_trial_coherence_power(
    r, j, config, fs, duration, sig_dict, out_dir, new_lfp
):
    coherence_res = calculate_banded_coherence(
        sig_dict["SUB"], sig_dict["RSC"], fs, config
    )
    if coherence_res is None:
        return None, None, None
    results_list, coherence_list, pxx_list = [], [], []

    for i, trial_type in zip(range(1, 3), ("forced", "choice")):
        lfp_portions, final_trial_type, group, time_dict = setup_recording_info(
            r, fs, duration, i, trial_type, config
        )
        for k, bounds in lfp_portions.items():
            res_list = []
            fn_params = [r, config, fs, sig_dict, final_trial_type, group, k, bounds]
            res = compute_coherence_per_trial(*fn_params, coherence_list, res_list)
            if res is not None:
                sub_lfp, rsc_lfp, f, Cxy = res
                res_list.extend(coherence_res)
                extract_decoding_vals(config, i, j, k, f, Cxy, new_lfp)
                compute_power_per_trial(*fn_params, pxx_list, res_list)
                res_list.extend(np.array(bounds) / fs)
                res_list.extend(np.array(time_dict[k]) / fs)
                results_list.append(res_list)

        if res is not None:
            sub_lfp, rsc_lfp, f, Cxy = res
            plot_results_intermittent(r, sub_lfp, rsc_lfp, f, Cxy, out_dir, fs)
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
    res = coherence_from_bounds(config, fs, sig_dict, bounds)
    if res is None:
        return
    sub_lfp, rsc_lfp, f, Cxy = res
    coherence_list.extend(
        make_coherence_tuple(r, final_trial_type, group, k, f_, cxy_)
        for f_, cxy_ in zip(f, Cxy)
    )
    res_list.extend(list_results(r, final_trial_type, k))
    res_list.extend(theta_beta(f, Cxy, config))
    res_list.append(group)
    res_list.append(r.attrs["RSC on target"])
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
        if np.sum(np.abs(lfp._samples)) < 0.1:
            res_dict[f"{region}-{k}_beta"] = np.nan
            res_dict[f"{region}-{k}_theta"] = np.nan
        else:
            bandpowers(config, res_dict, k, region, lfp)
    res_list += [
        res_dict[f"SUB-{k}_beta"],
        res_dict[f"SUB-{k}_theta"],
        res_dict[f"RSC-{k}_beta"],
        res_dict[f"RSC-{k}_theta"],
    ]
    f_welch, Pxx = compute_power(fs, sig_dict["SUB"], config)
    pxx_list.extend(
        make_power_tuple(r, final_trial_type, group, k, p_val, f_val)
        for p_val, f_val in zip(Pxx, f_welch)
    )


def make_power_tuple(r, final_trial_type, group, k, p_val, f_val):
    return [
        f_val,
        p_val,
        r.attrs["passed"],
        group,
        k,
        final_trial_type,
        r.attrs["RSC on target"],
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
        r.attrs["RSC on target"],
    )


def setup_recording_info(r, fs, duration, i, trial_type, config):
    max_lfp_lengths_seconds = config.get("max_lfp_lengths")
    time_dict = extract_times_for_lfp(r, fs, duration, i)
    lfp_portions = extract_lfp_portions(
        max_lfp_lengths_seconds, fs, duration, time_dict
    )
    final_trial_type = convert_trial_type(r, trial_type)
    group = get_group(r)
    return lfp_portions, final_trial_type, group, time_dict


def compute_power(fs, x, config):
    f_welch, Pxx = welch(
        x,
        fs=fs,
        nperseg=config["tmaze_winsec"] * fs,
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


def plot_results_intermittent(r, x, y, f, Cxy, out_dir, fs):
    every_few_iters = random()
    if every_few_iters < 0.05:
        fig2, ax2 = plt.subplots(3, 1)
        ax2[0].plot(f, Cxy, c="k")
        ax2[1].plot([i / fs for i in range(len(x))], x, c="k")
        ax2[2].plot([i / fs for i in range(len(y))], y, c="k")
        fig2.savefig(
            out_dir
            / f'coherence_{r.attrs["rat"]}_{r.attrs["session"]}_{r.attrs["trial"]}.png'
        )

        plt.close(fig2)


def get_group(r):
    return "Control" if r.attrs["rat"].lower().startswith("c") else "Lesion (ATNx)"


def save_computed_info(results_list, coherence_df_list, pxx_list, out_dir):
    headers = get_result_headers()
    res_df = list_to_df(results_list, headers=headers)
    df_to_file(res_df, out_dir / "results.csv", index=False)

    headers = get_coherence_headers()
    coherence_df = list_to_df(coherence_df_list, headers=headers)
    df_to_file(coherence_df, out_dir / "coherence.csv", index=False)

    headers = get_power_headers()
    power_df = list_to_df(pxx_list, headers)
    df_to_file(power_df, out_dir / "power.csv", index=False)


def save_decoding_values(out_dir, groups, choices, new_lfp):
    decoding_loc = out_dir / "decoding.csv"
    with open(decoding_loc, "w") as f:
        for i in range(len(groups)):
            line = ""
            line += f"{groups[i]},"
            line += f"{choices[i]},"
            for v in new_lfp[i]:
                line += f"{v},"
            line = line[:-1] + "\n"
            f.write(line)


def get_power_headers():
    return [
        "Frequency (Hz)",
        "Power (dB)",
        "Passed",
        "Group",
        "Part",
        "Trial",
        "RSC on target",
    ]


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
        "RSC on target",
    ]


def get_result_headers():
    return [
        "location",
        "session",
        "animal",
        "test",
        "choice",
        "part",
        "trial",
        "Theta Coherence",
        "Beta Coherence",
        "Peak Theta Coherence",
        "Group",
        "RSC on target",
        "Full Theta Coherence",
        "Full Belta Coherence",
        "SUB Beta Power",
        "SUB Theta Power",
        "RSC Beta Power",
        "RSC Theta Power",
        "LFP t1",
        "LFP t2",
        "t1",
        "t2",
        "t3",
    ]


def list_results(r, final_trial_type, k):
    res_list = [
        r.source_file,
        r.attrs["session"],
        r.attrs["rat"],
        r.attrs["trial"],
        r.attrs["passed"],
        k,
        final_trial_type,
    ]

    return res_list


def convert_trial_type(r, trial_type):
    if trial_type == "forced":
        return "Forced"
    elif r.attrs["passed"].strip().upper() == "Y":
        return "Correct"
    elif r.attrs["passed"].strip().upper() == "N":
        return "Incorrect"
    else:
        return "ERROR IN ANALYSIS"


def theta_beta(f, Cxy, config):
    theta_co = Cxy[np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))]
    beta_co = Cxy[np.nonzero((f >= config["beta_min"]) & (f <= config["beta_max"]))]
    mean_theta_coherence = np.nanmean(theta_co)
    mean_beta_coherence = np.nanmean(beta_co)

    theta_co_peak = Cxy[
        np.nonzero((f >= config["theta_min"]) & (f <= (config["theta_max"] + 0.5)))
    ]
    peak_theta_coherence = np.nanmax(theta_co_peak)
    return mean_theta_coherence, mean_beta_coherence, peak_theta_coherence


def extract_decoding_vals(config, i, j, k, f, Cxy, new_lfp):
    method = "coherence"
    if method == "coherence":
        if k == "choice":
            coherence_vals_for_decode = Cxy[
                np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))
            ]
            hf = new_lfp.shape[1] // 2
            s, e = (i - 1) * hf, i * hf
            new_lfp[j, s:e] = coherence_vals_for_decode


def coherence_from_bounds(config, fs, sig_dict, bounds):
    sub_s = sig_dict["SUB"]
    rsc_s = sig_dict["RSC"]
    x = np.array(sub_s[bounds[0] : bounds[1]])
    y = np.array(rsc_s[bounds[0] : bounds[1]])
    if (np.sum(np.abs(y)) < 0.1) or (np.sum(np.abs(x)) < 0.1):
        return None

    f, Cxy = coherence(x, y, fs, nperseg=config["tmaze_winsec"] * fs, nfft=256)
    f = f[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    Cxy = Cxy[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    return x, y, f, Cxy


def bandpowers(config, res_dict, k, region, lfp):
    beta_power = lfp.bandpower(
        [config["beta_min"], config["beta_max"]],
        window_sec=config["tmaze_winsec"],
        band_total=True,
    )
    theta_power = lfp.bandpower(
        [config["theta_min"], config["theta_max"]],
        window_sec=config["tmaze_winsec"],
        band_total=True,
    )
    res_dict[f"{region}-{k}_beta"] = beta_power["relative_power"]
    res_dict[f"{region}-{k}_theta"] = theta_power["relative_power"]


def extract_lfp_portions(max_lfp_lengths_seconds, fs, duration, time_dict):
    lfp_portions = {}
    for k, max_len in max_lfp_lengths_seconds.items():
        extract_start_choice_end(
            max_lfp_lengths_seconds, fs, time_dict, k, max_len, lfp_portions
        )
    verify_start_end(fs, duration, lfp_portions)
    return lfp_portions


def convert_signal_to_nc(bounds, signal, fs):
    lfp_t1, lfp_t2 = bounds
    lfp = NLfp()
    lfp._samples = np.array(signal[lfp_t1:lfp_t2])
    lfp._timestamp = np.array(list(range(len(lfp._samples)))) / fs
    lfp._record_info["Sampling rate"] = fs
    return lfp


def verify_start_end(fs, duration, lfp_portions):
    """Make sure have at least 1 second and not > duration."""
    for k, v in lfp_portions.items():
        start_time, end_time = v
        if (end_time - start_time) < fs:
            end_time = ceil(start_time + fs)

        if end_time > int(ceil(duration * 250)):
            end_time = int(floor(duration * 250))
        if start_time < 0:
            start_time = 0

        lfp_portions[k] = [start_time, end_time]


def extract_start_choice_end(
    max_lfp_lengths_seconds, fs, time_dict, k, max_len, lfp_portions
):
    """Find start time, choice time, and end time"""
    start_time = time_dict[k][0]
    choice_time = time_dict[k][1]
    end_time = time_dict[k][2]

    if k == "start":
        max_ch = (lfp_portions["choice"][1] - lfp_portions["choice"][0]) / fs
        ct = max_lfp_lengths_seconds["choice"][0]
        start_time, end_time = extract_first_times(ct, fs, max_ch, start_time, end_time)
    elif k == "choice":
        ct = max_lfp_lengths_seconds["choice"]
        start_time, end_time = extract_choice_times(
            ct, fs, start_time, choice_time, end_time
        )
    elif k == "end":
        max_ch = (lfp_portions["choice"][1] - lfp_portions["choice"][0]) / fs
        ct = max_lfp_lengths_seconds["choice"][1]
        start_time, end_time = extract_end_times(ct, fs, max_ch, start_time, end_time)
    else:
        raise RuntimeError(f"Unsupported key {k}")
    lfp_portions[k] = [floor(start_time), ceil(end_time)]


def extract_first_times(ct, fs, max_len, start_time, end_time):
    """
    Get start times.

    If the start bit is longer than max_len, take the last X
    seconds before the choice data
    """
    end_time = end_time - int(floor(ct * fs))
    start_time = max(0, end_time - (max_len * fs))
    return start_time, end_time


def extract_choice_times(ct, fs, start_time, choice_time, end_time):
    """
    Get choice times.

    For the choice, take (max_len[0], max_len[1]) seconds
    of data around the point.
    """
    left_push = int(floor(ct[0] * fs))
    right_push = int(ceil(ct[1] * fs))

    start_time = choice_time - left_push
    end_time = choice_time + right_push
    return start_time, end_time


def extract_end_times(ct, fs, max_len, start_time, end_time):
    """
    Get end times.

    For the end time, if the end is longer than max_len
    take the first X seconds after the choice data
    """
    start_time = start_time + int(ceil(ct * fs))
    end_time = start_time + (max_len * fs)
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

    return {
        "start": (lfpt1, lfpt2, lfpt2),
        "choice": (lfpt1, lfpt2, lfpt3),
        "end": (lfpt2, lfpt3, lfpt3),
    }


def extract_lfp_info(r):
    sub_lfp = r.data.processing["average_lfp"]["SUB_avg"].data[:]
    rsc_lfp = r.data.processing["average_lfp"]["RSC_avg"].data[:]
    fs = r.data.processing["average_lfp"]["SUB_avg"].rate
    duration = len(sub_lfp) / fs
    return sub_lfp, rsc_lfp, fs, duration


def calculate_banded_coherence(x, y, fs, config):
    if (np.sum(np.abs(x)) < 0.1) or (np.sum(np.abs(y)) < 0.1):
        return None
    f, Cxy = coherence(x, y, fs, nperseg=config["tmaze_winsec"] * fs)
    f = f[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]
    Cxy = Cxy[np.nonzero((f >= config["tmaze_minf"]) & (f <= config["tmaze_maxf"]))]

    theta_co = Cxy[np.nonzero((f >= config["theta_min"]) & (f <= config["theta_max"]))]
    beta_co = Cxy[np.nonzero((f >= config["beta_min"]) & (f <= config["beta_max"]))]
    theta_coherence = np.nanmean(theta_co)
    beta_coherence = np.nanmean(beta_co)
    return theta_coherence, beta_coherence


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]).parent,
    )
