from math import floor, ceil
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skm_pyutils.py_table import list_to_df, df_to_file
from skm_pyutils.py_plot import UnicodeGrabber
import simuran
import pandas as pd
import scipy.signal
import scipy.integrate

from lfp_atn_simuran.Scripts.lfp_clean import LFPClean


# 3. Compare theta and speed
def speed_vs_amp(self, lfp_signal, low_f, high_f, filter_kwargs=None, **kwargs):
    """Self represents an nc_spatial object."""
    lim = kwargs.get("range", [0, self.get_duration()])
    samples_per_sec = kwargs.get("samplesPerSec", 5)
    do_spectrogram_plot = kwargs.get("do_spectogram_plot", False)
    do_once = True

    # Not filtering anymore since using relative power
    # if filter_kwargs is None:
    #     filter_kwargs = {}
    # try:
    #     lfp_signal = lfp_signal.filter(low_f, high_f, **filter_kwargs)
    # except BaseException:
    #     lfp_signal = deepcopy(lfp_signal)
    #     _filt = [10, low_f, high_f, "bandpass"]
    #     lfp_signal._set_samples(
    #         butter_filter(
    #             lfp_signal.get_samples(), lfp_signal.get_sampling_rate(), *_filt
    #         )
    #     )

    # Calculate the LFP power
    skip_rate = int(self.get_sampling_rate() / samples_per_sec)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    index_to_grab = np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
    time_to_use = self.get_time()[index_to_grab][slicer]
    speed = self.get_speed()

    avg_speed = np.zeros_like(time_to_use)
    lfp_amplitudes = np.zeros_like(time_to_use)
    lfp_samples = lfp_signal.get_samples()
    if hasattr(lfp_samples, "unit"):
        import astropy.units as u

        lfp_samples = lfp_samples.to(u.uV).value
    else:
        lfp_samples = lfp_samples * 1000

    for i, t in enumerate(time_to_use):
        diff = 1 / (2 * samples_per_sec)

        low_sample = floor((t - diff) * self.get_sampling_rate())
        high_sample = floor((t + diff) * self.get_sampling_rate())
        avg_speed[i] = np.mean(speed[low_sample:high_sample])

        low_sample = floor((t - diff) * lfp_signal.get_sampling_rate())
        high_sample = ceil((t + diff) * lfp_signal.get_sampling_rate())
        if high_sample < len(lfp_samples):
            lfp_sample_200ms = lfp_samples[low_sample : high_sample + 1]
            slep_win = scipy.signal.hann(lfp_sample_200ms.size, False)
            f, psd = scipy.signal.welch(
                lfp_sample_200ms,
                fs=lfp_signal.get_sampling_rate(),
                window=slep_win,
                nperseg=len(lfp_sample_200ms),
                nfft=256,
                noverlap=0,
            )
            idx_band = np.logical_and(f >= low_f, f <= high_f)
            abs_power = scipy.integrate.simps(psd[idx_band], x=f[idx_band])
            total_power = scipy.integrate.simps(psd, x=f)
            lfp_amplitudes[i] = abs_power / total_power
        elif do_once:
            simuran.log.warning(
                "Position data ({}s) is longer than EEG data ({}s)".format(
                    time_to_use[-1], len(lfp_samples) / lfp_signal.get_sampling_rate()
                )
            )
            do_once = False

    min_speed, max_speed = kwargs.get("SpeedRange", [0, 40])
    # This kind of logic can be used in general to bin the speeds
    # binsize = kwargs.get("SpeedBinSize", 2)

    # max_speed = min(max_speed, np.ceil(avg_speed.max()))
    # min_speed = max(min_speed, np.floor(avg_speed.min()))
    # bins = np.arange(min_speed, max_speed, binsize)

    # visit_time = np.histogram(avg_speed, bins)[0]
    # speedInd = np.digitize(avg_speed, bins) - 1

    # binned_lfp = [np.sum(lfp_amplitudes[speedInd == i]) for i in range(len(bins) - 1)]
    # rate = non_zero_divide(np.array(binned_lfp), visit_time)
    # fig, ax = plt.subplots()
    # ax.plot(bins[:-1], rate)
    # plt.show()
    fig = None
    if do_spectrogram_plot:
        spectrogram_len = min(
            kwargs.get("SpectrogramLen", 150),
            len(lfp_samples) / lfp_signal.get_sampling_rate(),
        )
        high_sample = floor(spectrogram_len * lfp_signal.get_sampling_rate())
        high_speed_sample = floor(spectrogram_len * samples_per_sec)
        lfp_sample_to_use = lfp_samples[:high_sample]
        f, t, Sxx = scipy.signal.spectrogram(
            x=lfp_sample_to_use,
            fs=lfp_signal.get_sampling_rate(),
            nperseg=(lfp_signal.get_sampling_rate() // 2),
            nfft=512,
        )
        Sxx = 10 * np.log10(Sxx / np.amax(Sxx))
        Sxx = Sxx.flatten()
        Sxx[np.nonzero(Sxx < -40)] = -40
        Sxx = np.reshape(Sxx, [f.size, t.size])
        fig, ax = plt.subplots(figsize=(16, 10))

        c_map = "magma"
        levels = 22
        dx = np.mean(np.diff(t))
        dy = np.mean(np.diff(f))
        pad_map = np.pad(Sxx[:-1, :-1], ((1, 1), (1, 1)), "edge")
        vmin, vmax = np.nanmin(pad_map), np.nanmax(pad_map)
        if vmax - vmin > 0.1:
            splits = np.linspace(vmin, vmax, levels + 1)
        else:
            splits = np.linspace(vmin, vmin + 0.1 * levels, levels + 1)
        splits = np.around(splits, decimals=1)
        to_delete = []
        for i in range(len(splits) - 1):
            if splits[i] >= splits[i + 1]:
                to_delete.append(i)
        splits = np.delete(splits, to_delete)
        x_edges = np.append(t - dx / 2, t[-1] + dx / 2)
        y_edges = np.append(f - dy / 2, f[-1] + dy / 2)
        pcm = ax.contourf(
            x_edges, y_edges, pad_map, levels=splits, cmap=c_map, corner_mask=True
        )

        # pcm = ax.pcolormesh(
        #     t, f, Sxx, cmap=c_map, edgecolors="none", rasterized=True, shading="auto"
        # )
        ax.set_xlim(t.min(), t.max())
        ax.set_ylim(0, 30)
        simuran.despine()

        plt.colorbar(pcm, ax=ax, use_gridspec=True)

        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Frequency (Hz)")

        # ax2 = ax.twinx()
        ax.plot(
            time_to_use[:high_speed_sample],
            0.25 * avg_speed[:high_speed_sample],
            c="k",
            alpha=1,
            linewidth=3,
        )
        # ax2.set_ylabel("Speed (cm / s)")
        # ax2.set_ylim(0, np.max(avg_speed))

    pd_df = list_to_df(
        [avg_speed, lfp_amplitudes], transpose=True, headers=["Speed", "LFP amplitude"]
    )
    pd_df = pd_df[pd_df["Speed"] <= max_speed]
    pd_df["RoundedSpeed"] = np.around(pd_df["Speed"])

    return pd_df, fig


def speed_lfp_amp(
    recording,
    figures,
    base_dir,
    clean_method="avg",
    fmin=5,
    fmax=12,
    speed_sr=10,
    **kwargs,
):
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    signals_grouped_by_region = lc.clean(
        recording.signals, 0.5, 100, method_kwargs=clean_kwargs
    )["signals"]
    fmt = kwargs.get("image_format", "png")
    do_spectrogram_plot = kwargs.get("do_spectogram_plot", False)

    # Single values
    spatial = recording.spatial.underlying
    simuran.set_plot_style()
    results = {}
    skip_rate = int(spatial.get_sampling_rate() / speed_sr)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    speed = spatial.get_speed()[slicer]
    results["mean_speed"] = np.mean(speed)
    results["duration"] = spatial.get_duration()
    results["distance"] = results["mean_speed"] * results["duration"]

    basename = recording.get_name_for_save(base_dir)

    # Figures
    simuran.set_plot_style()
    for name, signal in signals_grouped_by_region.items():
        lfp_signal = signal

        # Speed vs LFP power
        pd_df, sfig = speed_vs_amp(
            spatial,
            lfp_signal,
            fmin,
            fmax,
            samplesPerSec=speed_sr,
            do_spectrogram_plot=do_spectrogram_plot,
        )

        if do_spectrogram_plot:
            out_name = basename + "_speed_theta_spectogram_{}".format(name)
            sfig = simuran.SimuranFigure(sfig, out_name, dpi=400, done=True, format=fmt)
            figures.append(sfig)

        results[f"{name}_df"] = pd_df

        fig, ax = plt.subplots()
        sns.lineplot(data=pd_df, x="RoundedSpeed", y="LFP amplitude", ax=ax)
        simuran.despine()
        fname = basename + "_speed_theta_corr_{}".format(name)
        speed_amp_fig = simuran.SimuranFigure(
            fig, filename=fname, done=True, format=fmt, dpi=400
        )
        figures.append(speed_amp_fig)

    return results


def define_recording_group(base_dir, main_dir):
    dirs = base_dir[len(main_dir + os.sep) :].split(os.sep)
    dir_to_check = dirs[0]
    if dir_to_check.startswith("CS"):
        group = "Control"
    elif dir_to_check.startswith("LS"):
        group = "Lesion"
    else:
        group = "Undefined"

    number = int(dir_to_check.split("_")[0][-1])
    return group, number


def combine_results(info, extra_info, **kwargs):
    """This uses the pickle output from SIMURAN."""
    simuran.set_plot_style()
    data_animal_list, fname_animal_list = info
    out_dir, name = extra_info
    os.makedirs(out_dir, exist_ok=True)

    n_ctrl_animals = 0
    n_lesion_animals = 0
    df_lists = []
    for item_list, fname_list in zip(data_animal_list, fname_animal_list):
        r_ctrl = 0
        r_les = 0
        for item_dict, fname in zip(item_list, fname_list):
            item_dict = item_dict["speed_lfp_amp"]
            data_set, number = define_recording_group(
                os.path.dirname(fname), kwargs["cfg_base_dir"]
            )

            # if number >= 4:
            #     continue

            # Skip LSR7 if present
            if number == 7:
                continue

            if data_set == "Control":
                r_ctrl += 1
            else:
                r_les += 1

            for r in ["SUB", "RSC"]:
                id_ = item_dict[r + "_df"]
                id_["Group"] = data_set
                id_["region"] = r
                id_["number"] = number
                df_lists.append(id_)

            # ic(
            #     fname,
            #     data_set,
            #     number,
            #     item_dict["mean_speed"],
            #     len(item_dict["RSC_df"]),
            # )
        n_ctrl_animals += r_ctrl / len(fname_list)
        n_lesion_animals += r_les / len(fname_list)

    simuran.print(f"{n_ctrl_animals} CTRL animals, {n_lesion_animals} Lesion animals")

    out_dfname = os.path.join(out_dir, "summary", "speed_results.csv")
    df = pd.concat(df_lists, ignore_index=True)
    df_to_file(df, out_dfname)

    df.replace("Control", f"Control", inplace=True)
    df.replace("Lesion", f"ATNx (Lesion)", inplace=True)

    simuran.print("Saving plots to {}".format(os.path.join(out_dir, "summary")))

    control_df = df[df["Group"] == f"ATNx (Lesion)"]
    sub_df = control_df[control_df["region"] == "RSC"]
    simuran.print(sub_df.groupby("RoundedSpeed").mean())
    for ci, oname in zip([95, None], ["_ci", ""]):
        sns.lineplot(
            data=df[df["region"] == "SUB"],
            x="RoundedSpeed",
            y="LFP amplitude",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
            # estimator="mean",
        )
        simuran.despine()
        plt.xlabel("Speed (cm / s)")
        plt.ylabel("Amplitude ({}V)".format(UnicodeGrabber.get("micro")))
        plt.title("Subicular LFP power (median)")

        os.makedirs(os.path.join(out_dir, "summary"), exist_ok=True)
        plt.savefig(
            os.path.join(
                out_dir, "summary", name + "--sub--speed--theta{}.pdf".format(oname)
            ),
            dpi=400,
        )

        plt.close("all")

        sns.lineplot(
            data=df[df["region"] == "RSC"],
            x="RoundedSpeed",
            y="LFP amplitude",
            style="Group",
            hue="Group",
            ci=ci,
            estimator=np.median,
            # estimator="mean",
        )
        simuran.despine()
        plt.xlabel("Speed (cm / s)")
        plt.ylabel("Amplitude ({}V)".format(UnicodeGrabber.get("micro")))
        plt.title("Retrosplenial LFP power (median)")

        plt.savefig(
            os.path.join(
                out_dir, "summary", name + "--rsc--speed--theta{}.pdf".format(oname)
            ),
            dpi=400,
        )

        plt.close("all")
