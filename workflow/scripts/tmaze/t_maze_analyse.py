import os
from math import floor, ceil
from pprint import pprint
import csv
import argparse

import simuran
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

try:
    from lfp_atn_simuran.Scripts.lfp_clean import LFPClean

    do_analysis = True
except ImportError:
    do_analysis = False

from neuronal.decoding import LFPDecoder


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
    excel_location,
    base_dir,
    plot_individual_sessions,
    do_coherence=True,
    do_decoding=True,
    overwrite=False,
):

    # Setup
    df = df_from_file(excel_location)
    cfg = simuran.parse_config()
    delta_min = cfg["delta_min"]
    delta_max = cfg["delta_max"]
    theta_min = cfg["theta_min"]
    theta_max = cfg["theta_max"]
    clean_method = cfg["clean_method"]
    clean_kwargs = cfg["clean_kwargs"]

    window_sec = 0.5
    fmin, fmax = 2.0, 40
    max_lfp_lengths_seconds = {"start": 20, "choice": (3.5, 0.5), "end": 10}

    ituples = df.itertuples()
    num_rows = len(df)

    no_pass = False
    if "passed" not in df.columns:
        print("Please add passed as a column to the df.")
        no_pass = True

    results = []
    coherence_df_list = []

    base_dir_new = os.path.dirname(excel_location)
    here = os.path.dirname(os.path.abspath(__file__))
    decoding_loc = out_name = os.path.join(
        here, "..", "sim_results", "tmaze", "lfp_decoding.csv"
    )
    lfp_len = 6
    hf = lfp_len // 2
    new_lfp = np.zeros(shape=(num_rows // 2, lfp_len))
    groups = []
    choices = []
    pxx_arr = []
    oname_coherence = os.path.join(
        here, "..", "sim_results", "tmaze", "coherence_full.csv"
    )
    oname_power_tmaze = os.path.join(
        here, "..", "sim_results", "tmaze", "power_tmaze_full.csv"
    )
    split = os.path.splitext(os.path.basename(excel_location))
    o_name_res = os.path.join(
        here, "..", "sim_results", "tmaze", split[0] + "_results" + split[1]
    )

    # Load existing data if instructed to and it exists
    os.makedirs(os.path.dirname(oname_coherence), exist_ok=True)
    skip = (
        (os.path.exists(decoding_loc))
        and (not overwrite)
        and (os.path.exists(oname_coherence))
    )
    if skip:
        with open(decoding_loc, "r") as f:
            csvreader = csv.reader(f, delimiter=",")
            for i, row in enumerate(csvreader):
                groups.append(row[0])
                choices.append(row[1])
                vals = row[2:]
                new_lfp[i] = np.array([float(v) for v in vals[:lfp_len]])

        coherence_df = df_from_file(oname_coherence)
        power_df = df_from_file(oname_power_tmaze)
        res_df = df_from_file(o_name_res)

    ## Extract LFP, do coherence, and plot
    if not skip:
        for j in range(num_rows // 2):

            # row1 is the forced movement
            row1 = next(ituples)
            # row2 is the choice trial
            row2 = next(ituples)

            # Load the t-maze data
            recording_location = os.path.join(base_dir, row1.location)
            recording_location = recording_location.replace("--", os.sep)
            param_file = os.path.join(here, "..", "recording_mappings", row1.mapping)
            recording = simuran.Recording(
                param_file=param_file, base_file=recording_location, load=False
            )
            lfp_clean = LFPClean(method=clean_method, visualise=False)
            recording.signals.load()
            sig_dict = lfp_clean.clean(
                recording, min_f=fmin, max_f=fmax, method_kwargs=clean_kwargs
            )["signals"]
            x = np.array(sig_dict["SUB"].samples.to(u.mV))
            duration = len(x) / 250
            y = np.array(sig_dict["RSC"].samples.to(u.mV))
            fig, ax = plt.subplots()

            fs = sig_dict["SUB"].sampling_rate

            # Setup and loading done -- Analyse the t-maze data
            if do_coherence:
                # Coherence over the whole recording
                f, Cxy = coherence(x, y, fs, nperseg=window_sec * 250)
                f = f[np.nonzero((f >= fmin) & (f <= fmax))]
                Cxy = Cxy[np.nonzero((f >= fmin) & (f <= fmax))]

                theta_co = Cxy[np.nonzero((f >= theta_min) & (f <= theta_max))]
                delta_co = Cxy[np.nonzero((f >= delta_min) & (f <= delta_max))]
                max_theta_coherence_ = np.nanmean(theta_co)
                max_delta_coherence_ = np.nanmean(delta_co)

            if plot_individual_sessions:
                # Used to plot t-maze sessions - mostly for verification
                recording.spatial.load()
                spatial = recording.spatial.underlying
                fig, ax = plt.subplots()

            for k_, r in enumerate(
                (
                    row1,
                    row2,
                )
            ):
                if k_ == 0:
                    trial_type = "forced"
                else:
                    trial_type = "choice"

                # Parse out the times
                t1, t2, t3 = r.start, r.choice, r.end

                # Make sure there are no parsing Incorrect
                if t3 > duration:
                    raise RuntimeError(
                        "Last time {} greater than duration {}".format(t3, duration)
                    )

                # Convert these times into LFP samples
                lfpt1, lfpt2, lfpt3 = (
                    int(floor(t1 * fs)),
                    int(ceil(t2 * fs)),
                    int(ceil(t3 * fs)),
                )

                # Split the LFP into three parts, the start, choice, and end
                lfp_portions = {}
                time_dict = {
                    "start": (lfpt1, lfpt2, lfpt2),
                    "choice": (lfpt1, lfpt2, lfpt3),
                    "end": (lfpt2, lfpt3, lfpt3),
                }
                for k, v in max_lfp_lengths_seconds.items():
                    max_len = v
                    start_time = time_dict[k][0]
                    choice_time = time_dict[k][1]
                    end_time = time_dict[k][2]

                    if k == "start":
                        # If the start bit is longer than max_len, take the last X
                        # seconds before the choice data
                        ct = max_lfp_lengths_seconds["choice"][0]
                        end_time = max(end_time - int(floor(ct * fs)), start_time)
                        natural_start_time = end_time - max_len * fs
                        start_time = max(natural_start_time, start_time)
                    elif k == "choice":
                        # For the choice, take (max_len[0], max_len[1]) seconds
                        # of data around the point
                        left_push = int(floor(v[0] * fs))
                        right_push = int(ceil(v[1] * fs))

                        start_time = max(choice_time - left_push, start_time)
                        end_time = min(choice_time + right_push, end_time)
                    elif k == "end":
                        # For the end time, if the end is longer than max_len, take the first X seconds after the choice data
                        ct = max_lfp_lengths_seconds["choice"][1]
                        start_time = min(start_time + int(ceil(ct * fs)), end_time)
                        natural_end_time = start_time + max_len * fs
                        end_time = min(natural_end_time, end_time)
                    else:
                        raise RuntimeError(f"Unsupported key {k}")

                    # Make sure have at least 1 second
                    if (end_time - start_time) < fs:
                        end_time = start_time + fs

                    if end_time > int(ceil(duration * 250)):
                        raise RuntimeError(
                            "End time {} greater than duration {}".format(
                                end_time, duration
                            )
                        )

                    lfp_portions[k] = (start_time, end_time)

                if j % 20 == 0:
                    print(f"On iteration {j} of {num_rows // 2} -- trial {trial_type}")
                    for k in lfp_portions.keys():
                        print(
                            "{}: {} -- {}".format(
                                k,
                                np.array(time_dict[k]) / 250,
                                np.array(lfp_portions[k]) / 250,
                            )
                        )
                    print("----------------------")

                if plot_individual_sessions:
                    if r.test == "first":
                        c = "k"
                    else:
                        c = "r"

                    st1, st2 = int(floor(t1 * 50)), int(ceil(t3 * 50))
                    x_time = spatial.get_pos_x()[st1:st2]
                    y_time = spatial.get_pos_y()[st1:st2]
                    c_end = int(floor(t2 * 50))
                    spat_c = (spatial.get_pos_x()[c_end], spatial.get_pos_y()[c_end])
                    ax.plot(x_time, y_time, c=c, label=r.test)
                    ax.plot(spat_c[0], spat_c[1], c="b", marker="x", label="decision")
                    ax.plot(x_time[0], y_time[0], c="b", marker="o", label="start")
                    ax.plot(x_time[-1], y_time[-1], c="b", marker=".", label="end")

                if do_coherence:
                    res_dict = {}
                    # Power
                    for k, v in lfp_portions.items():
                        for region, signal in sig_dict.items():
                            lfpt1, lfpt2 = v
                            lfp = NLfp()
                            lfp.set_channel_id(signal.channel)

                            lfp._timestamp = np.array(
                                signal.timestamps[lfpt1:lfpt2].to(u.s)
                            )
                            lfp._samples = np.array(
                                signal.samples[lfpt1:lfpt2].to(u.mV)
                            )
                            lfp._record_info["Sampling rate"] = signal.sampling_rate
                            delta_power = lfp.bandpower(
                                [delta_min, delta_max],
                                window_sec=window_sec,
                                band_total=True,
                            )
                            theta_power = lfp.bandpower(
                                [theta_min, theta_max],
                                window_sec=window_sec,
                                band_total=True,
                            )
                            res_dict["{}-{}_delta".format(region, k)] = delta_power[
                                "relative_power"
                            ]
                            res_dict["{}-{}_theta".format(region, k)] = theta_power[
                                "relative_power"
                            ]

                        sub_s = sig_dict["SUB"]
                        rsc_s = sig_dict["RSC"]
                        x = np.array(sub_s.samples[lfpt1:lfpt2].to(u.mV))
                        y = np.array(rsc_s.samples[lfpt1:lfpt2].to(u.mV))

                        f, Cxy = coherence(x, y, fs, nperseg=window_sec * 250, nfft=256)
                        f = f[np.nonzero((f >= fmin) & (f <= fmax))]
                        Cxy = Cxy[np.nonzero((f >= fmin) & (f <= fmax))]

                        if do_decoding:
                            if k == "choice":
                                coherence_vals_for_decode = Cxy[
                                    np.nonzero((f >= theta_min) & (f <= theta_max))
                                ]
                                s, e = (k_) * hf, (k_ + 1) * hf
                                new_lfp[j, s:e] = coherence_vals_for_decode

                        theta_co = Cxy[np.nonzero((f >= theta_min) & (f <= theta_max))]
                        delta_co = Cxy[np.nonzero((f >= delta_min) & (f <= delta_max))]
                        max_theta_coherence = np.nanmean(theta_co)
                        max_delta_coherence = np.nanmean(delta_co)

                        theta_co_peak = Cxy[np.nonzero((f >= 11.0) & (f <= 13.0))]
                        peak_theta_coherence = np.nanmax(theta_co_peak)

                        if trial_type == "forced":
                            final_trial_type = "Forced"
                        else:
                            if r.passed.strip().upper() == "Y":
                                final_trial_type = "Correct"
                            elif r.passed.strip().upper() == "N":
                                final_trial_type = "Incorrect"
                            else:
                                final_trial_type = "ERROR IN ANALYSIS"

                        res_list = [
                            r.location,
                            r.session,
                            r.animal,
                            r.test,
                            r.passed.strip(),
                            k,
                            final_trial_type,
                        ]
                        res_list += [
                            res_dict[f"SUB-{k}_delta"],
                            res_dict[f"SUB-{k}_theta"],
                            res_dict[f"RSC-{k}_delta"],
                            res_dict[f"RSC-{k}_theta"],
                        ]
                        res_list += [max_theta_coherence, max_delta_coherence]
                        res_list += [
                            max_theta_coherence_,
                            max_delta_coherence_,
                            peak_theta_coherence,
                        ]

                        if no_pass is False:
                            group = (
                                "Control"
                                if r.animal.lower().startswith("c")
                                else "Lesion (ATNx)"
                            )
                            if do_coherence:
                                for f_, cxy_ in zip(f, Cxy):
                                    coherence_df_list.append(
                                        (
                                            f_,
                                            cxy_,
                                            r.passed.strip(),
                                            group,
                                            r.test,
                                            r.session,
                                            k,
                                            final_trial_type,
                                        )
                                    )

                                f_welch, Pxx = welch(
                                    x,
                                    fs=fs,
                                    nperseg=window_sec * 250,
                                    return_onesided=True,
                                    scaling="density",
                                    average="mean",
                                )

                                f_welch = f_welch[
                                    np.nonzero((f_welch >= fmin) & (f_welch <= fmax))
                                ]
                                Pxx = Pxx[
                                    np.nonzero((f_welch >= fmin) & (f_welch <= fmax))
                                ]

                                # Convert to full scale relative dB (so max at 0)
                                Pxx_max = np.max(Pxx)
                                Pxx = 10 * np.log10(Pxx / Pxx_max)
                                for p_val, f_val in zip(Pxx, f_welch):
                                    pxx_arr.append(
                                        [
                                            f_val,
                                            p_val,
                                            r.passed.strip(),
                                            group,
                                            k,
                                            final_trial_type,
                                        ]
                                    )
                        res_list += [group]
                        results.append(res_list)

                    name = os.path.splitext(r.location)[0]
                    if plot_individual_sessions:
                        fig2, ax2 = plt.subplots(3, 1)
                        ax2[0].plot(f, Cxy, c="k")
                        ax2[1].plot([i / 250 for i in range(len(x))], x, c="k")
                        ax2[2].plot([i / 250 for i in range(len(y))], y, c="k")
                        base_dir_new = os.path.dirname(excel_location)
                        fig2.savefig(
                            os.path.join(
                                base_dir_new,
                                "coherence_{}_{}_{}.png".format(
                                    row1.location, r.session, r.test
                                ),
                            )
                        )
                        plt.close(fig2)

            if do_decoding:
                groups.append(group)
                choices.append(str(r.passed).strip())

            if plot_individual_sessions:
                ax.invert_yaxis()
                ax.legend()
                base_dir_new = os.path.dirname(excel_location)
                figname = os.path.join(base_dir_new, name) + "_tmaze.png"
                fig.savefig(figname, dpi=400)
                plt.close(fig)

    if do_coherence and not skip:
        # Save the results
        headers = [
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

        res_df = pd.DataFrame(results, columns=headers)

        split = os.path.splitext(os.path.basename(excel_location))
        out_name = os.path.join(
            here, "..", "sim_results", "tmaze", split[0] + "_results" + split[1]
        )
        df_to_file(res_df, out_name, index=False)

        # Plot difference between pass and fail trials
        headers = [
            "Frequency (Hz)",
            "Coherence",
            "Passed",
            "Group",
            "Test",
            "Session",
            "Part",
            "Trial",
        ]
        coherence_df = list_to_df(coherence_df_list, headers=headers)

        df_to_file(coherence_df, oname_coherence, index=False)

        power_df = list_to_df(
            pxx_arr,
            headers=[
                "Frequency (Hz)",
                "Power (dB)",
                "Passed",
                "Group",
                "Part",
                "Trial",
            ],
        )

        df_to_file(power_df, oname_power_tmaze, index=False)

    if do_coherence or skip:

        simuran.set_plot_style()
        # res_df["ID"] = res_df["trial"] + "_" + res_df["part"]
        res_df = res_df[res_df["part"] == "choice"]
        sns.barplot(
            data=res_df,
            x="trial",
            y="Theta_coherence",
            hue="Group",
            estimator=np.median,
        )
        plt.xlabel("Trial result")
        plt.ylabel("Theta coherence")
        plt.tight_layout()
        plt.savefig(
            os.path.join(here, "..", "sim_results", "tmaze", f"bar--coherence.pdf"),
            dpi=400,
        )
        plt.close("all")

        sns.barplot(
            data=res_df,
            x="trial",
            y="Delta_coherence",
            hue="Group",
            estimator=np.median,
        )
        plt.xlabel("Trial result")
        plt.ylabel("Delta coherence")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                here, "..", "sim_results", "tmaze", f"bar--coherence--delta.pdf"
            ),
            dpi=400,
        )
        plt.close("all")

        for group in ("Control", "Lesion (ATNx)"):
            coherence_df_sub = coherence_df[coherence_df["Group"] == group]
            power_df_sub = power_df[power_df["Group"] == group]
            sns.lineplot(
                data=coherence_df_sub,
                x="Frequency (Hz)",
                y="Coherence",
                hue="Part",
                style="Trial",
                ci=None,
                estimator=np.median,
            )
            plt.ylim(0, 1)
            simuran.despine()
            plt.savefig(
                os.path.join(
                    here, "..", "sim_results", "tmaze", f"{group}--coherence.pdf"
                ),
                dpi=400,
            )
            plt.close("all")

            sns.lineplot(
                data=coherence_df_sub,
                x="Frequency (Hz)",
                y="Coherence",
                hue="Part",
                style="Trial",
                ci=95,
                estimator=np.median,
            )
            plt.ylim(0, 1)
            simuran.despine()
            plt.savefig(
                os.path.join(
                    here, "..", "sim_results", "tmaze", f"{group}--coherence_ci.pdf"
                ),
                dpi=400,
            )
            plt.close("all")

            sns.lineplot(
                data=power_df_sub,
                x="Frequency (Hz)",
                y="Power (dB)",
                hue="Part",
                style="Trial",
                ci=95,
                estimator=np.median,
            )
            plt.xlim(0, 40)
            simuran.despine()
            plt.savefig(
                os.path.join(
                    here, "..", "sim_results", "tmaze", f"{group}--power_ci.pdf"
                ),
                dpi=400,
            )
            plt.close("all")

        # Choice - lesion vs ctrl coherence when Incorrect and correct
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
        )
        simuran.despine()
        plt.savefig(
            os.path.join(here, "..", "sim_results", "tmaze", "choice_power_ci.pdf"),
            dpi=400,
        )
        plt.close("all")

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
        simuran.despine()
        plt.savefig(
            os.path.join(here, "..", "sim_results", "tmaze", "choice_coherence_ci.pdf"),
            dpi=400,
        )
        plt.close("all")

    # Try to decode pass and fail trials.
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

    if do_decoding:
        groups = np.array(groups)
        labels = np.array(choices)
        decoding(
            new_lfp, groups, labels, os.path.join(here, "..", "sim_results", "tmaze")
        )


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
