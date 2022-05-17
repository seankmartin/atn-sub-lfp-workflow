def do_coherence(info, extra_info, **kwargs):
    data, fnames = info
    kwargs["fnames"] = fnames
    out_dir, name = extra_info
    plot_all_lfp(data, out_dir, name, **kwargs)


def do_spectrum(info, extra_info, **kwargs):
    out_dir, name = extra_info
    plot_all_spectrum(info, out_dir, name, **kwargs)


def plot_all_spectrum(info, out_dir, name, **kwargs):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import simuran

    from neurochat.nc_utils import smooth_1d
    from skm_pyutils.py_plot import UnicodeGrabber
    from skm_pyutils.py_table import list_to_df
    from fooof import FOOOFGroup

    cfg = kwargs

    scale = cfg["psd_scale"]
    # base_dir = cfg["cfg_base_dir"]

    os.makedirs(out_dir, exist_ok=True)

    simuran.set_plot_style()

    # Control args
    smooth_power = False
    fmax = 40

    parsed_info = []
    info_for_fooof_ctrl = {
        "SUB": {"frequency": None, "spectra": []},
        "RSC": {"frequency": None, "spectra": []},
    }
    info_for_fooof_lesion = {
        "SUB": {"frequency": None, "spectra": []},
        "RSC": {"frequency": None, "spectra": []},
    }
    data, fnames = info
    n_ctrl_animals = 0
    n_lesion_animals = 0
    for item_list, fname_list in zip(data, fnames):
        r_ctrl = 0
        r_les = 0
        for item_dict, fname in zip(item_list, fname_list):
            # animal_name = fname[len(base_dir+os.sep):].split(os.sep)[0]
            # animal_name = animal_name.split("_")[0]
            # animal_number = int(animal_name[-1])
            # if animal_number >= 4:
            #     continue
            item_dict = item_dict["powers"]
            data_set = item_dict["SUB" + " welch"][2][0]
            if data_set == "Control":
                r_ctrl += 1
            else:
                r_les += 1

            for r in ["SUB", "RSC"]:
                id_ = item_dict[r + " welch"]
                powers = id_[1]
                if smooth_power:
                    powers = smooth_1d(
                        id_[1].astype(float),
                        kernel_type="hg",
                        kernel_size=5,
                    )

                if id_[2][0] == "Control":
                    info_for_fooof_ctrl[r]["frequency"] = id_[0]
                    if scale != "volts":
                        max_pxx = item_dict[r + " max f"]
                        volts_scale = (
                            np.power(10.0, (np.array(powers).astype(np.float64) / 10.0))
                            * max_pxx
                        )
                        info_for_fooof_ctrl[r]["spectra"].append(volts_scale)
                else:
                    info_for_fooof_lesion[r]["frequency"] = id_[0]
                    if scale != "volts":
                        max_pxx = item_dict[r + " max f"]
                        volts_scale = (
                            np.power(10.0, (np.array(powers).astype(np.float64) / 10.0))
                            * max_pxx
                        )
                        info_for_fooof_lesion[r]["spectra"].append(volts_scale)

                # Can change to volts if in DB
                # powers = volts_scale
                for v1, v2, v3, v4 in zip(id_[0], powers, id_[2], id_[3]):
                    if float(v1) < fmax:
                        parsed_info.append([v1, v2, v3, v4])

        n_ctrl_animals += r_ctrl / len(fname_list)
        n_lesion_animals += r_les / len(fname_list)
    print(f"{n_ctrl_animals} CTRL animals, {n_lesion_animals} Lesion animals")

    data = np.array(parsed_info)
    df = pd.DataFrame(data, columns=["frequency", "power", "Group", "region"])
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 5)", inplace=True)
    df[["frequency", "power"]] = df[["frequency", "power"]].apply(pd.to_numeric)

    print("Saving plots to {}".format(os.path.join(out_dir, "summary")))
    for ci, oname in zip([95, None], ["_ci", ""]):
        sns.lineplot(
            data=df[df["region"] == "SUB"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, fmax)
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            plt.ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            plt.ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.title("Subicular LFP power (median)")
        plt.tight_layout()

        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        plt.savefig(
            os.path.join(out_dir, "summary", name + "--sub--power{}.pdf".format(oname)),
            dpi=400,
        )

        plt.close("all")

        sns.lineplot(
            data=df[df["region"] == "RSC"],
            x="frequency",
            y="power",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
        )
        sns.despine(offset=0, trim=True)
        plt.xlabel("Frequency (Hz)")
        plt.xlim(0, fmax)
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            plt.ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            plt.ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.title("Retrosplenial LFP power (median)")
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, "summary", name + "--rsc--power{}.pdf".format(oname)),
            dpi=400,
        )

        plt.close("all")

    # FOOOF
    peaks_data = []
    for r in ["SUB", "RSC"]:
        fg = FOOOFGroup(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=2,
            min_peak_height=0.1,
            peak_threshold=2.0,
            aperiodic_mode="fixed",
        )

        fooof_arr_s = np.array(info_for_fooof_ctrl[r]["spectra"])
        fooof_arr_f = np.array(info_for_fooof_ctrl[r]["frequency"])
        fg.fit(fooof_arr_f, fooof_arr_s, [0.5, fmax], progress="tqdm")
        out_name = name + f"--{r}--foof--ctrl.pdf"
        fg.save_report(out_name, os.path.join(out_dir, "summary"))

        peaks = fg.get_params("peak_params", 0)[:, 0]

        for p in peaks:
            peaks_data.append([p, "Control", r])

        fg = FOOOFGroup(
            peak_width_limits=[1.0, 8.0],
            max_n_peaks=2,
            min_peak_height=0.1,
            peak_threshold=2.0,
            aperiodic_mode="fixed",
        )

        fooof_arr_s = np.array(info_for_fooof_lesion[r]["spectra"])
        fooof_arr_f = np.array(info_for_fooof_lesion[r]["frequency"])
        fg.fit(fooof_arr_f, fooof_arr_s, [0.5, fmax], progress="tqdm")
        out_name = name + f"--{r}--foof--lesion.pdf"
        fg.save_report(out_name, os.path.join(out_dir, "summary"))

        peaks = fg.get_params("peak_params", 0)[:, 0]

        for p in peaks:
            peaks_data.append([p, "ATNx (Lesion)", r])

    peaks_df = list_to_df(peaks_data, headers=["Peak frequency", "Group", "Region"])

    for r in ["SUB", "RSC"]:
        fig, ax = plt.subplots()
        sns.histplot(
            data=peaks_df[peaks_df["Region"] == r],
            x="Peak frequency",
            hue="Group",
            multiple="stack",
            # element="step",
            ax=ax,
            binwidth=1,
        )
        simuran.despine()
        ax.set_title(f"{r} Peak frequencies")
        ax.set_xlim(2, 20)
        out_name = os.path.join(out_dir, "summary", name + f"--foof--{r}combined.pdf")
        fig.savefig(out_name, dpi=400)


def plot_all_lfp(info, out_dir, name, **kwargs):
    import os

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import simuran
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    simuran.set_plot_style()

    parsed_info = []
    control_data = []
    lesion_data = []
    x_data = []
    for item, fnames in zip(info, kwargs["fnames"]):
        for val, fname in zip(item, fnames):
            # l1 = freq, l2 - coherence, l3 - group
            this_item = list(val.values())[0]
            to_use = this_item["full_res"]
            # to_use[1] = smooth_1d(
            #     this_item[1].astype(float), kernel_type="hg", kernel_size=5
            # )
            if to_use[2][0] == "Control":
                control_data.append(to_use[1])
            else:
                lesion_data.append(to_use[1])
            x_data = to_use[0].astype(float)
            parsed_info.append(np.array(to_use))

    lesion_arr = np.array(lesion_data).astype(float)
    control_arr = np.array(control_data).astype(float)

    y_lesion = np.median(lesion_arr, axis=0)
    y_control = np.median(control_arr, axis=0)

    difference = y_control[:80] - y_lesion[:80]

    data = np.concatenate(parsed_info, axis=1)
    df = pd.DataFrame(data.transpose(), columns=["frequency", "coherence", "Group"])
    df.replace("Control", "Control", inplace=True)
    df.replace("Lesion", "ATNx (Lesion)", inplace=True)
    df[["frequency", "coherence"]] = df[["frequency", "coherence"]].apply(pd.to_numeric)

    sns.lineplot(
        data=df[df["frequency"] <= 40],
        x="frequency",
        y="coherence",
        style="Group",
        hue="Group",
        estimator=np.median,
        ci=95,
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.ylim(0, 1)
    simuran.despine()

    print("Saving plots to {}".format(out_dir))
    os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "summary", name + "_ci.pdf"), dpi=400)
    plt.close("all")

    sns.lineplot(
        data=df[df["frequency"] <= 40],
        x="frequency",
        y="coherence",
        style="Group",
        hue="Group",
        estimator=np.median,
        ci=None,
    )
    plt.ylim(0, 1)

    simuran.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")

    print("Saving plots to {}".format(os.path.join(out_dir, "summary")))
    plt.savefig(os.path.join(out_dir, "summary", name + ".pdf"), dpi=400)

    plt.ylim(0, 1)

    plt.savefig(
        os.path.join(out_dir, "summary", name + "_full.pdf"), dpi=400
    )
    plt.close("all")

    sns.set_style("ticks")
    sns.set_palette("colorblind")

    sns.lineplot(x=x_data[:80], y=difference)

    simuran.despine()

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference")

    plt.savefig(os.path.join(out_dir, "summary", name + "--difference.pdf"), dpi=400)
    plt.close("all")
