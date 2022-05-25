import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran
from astropy import units as u
from scipy.signal import welch
from skm_pyutils.plot import UnicodeGrabber

from .lfp_clean import LFPClean


def calculate_psd(x, fs=250, fmin=1, fmax=100, scale="volts"):
    f, Pxx = welch(
        x.samples.to(u.uV).value,
        fs=fs,
        nperseg=2 * fs,
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    f = f[np.nonzero((f >= fmin) & (f <= fmax))]
    Pxx = Pxx[np.nonzero((f >= fmin) & (f <= fmax))]
    Pxx_max = np.max(Pxx)
    if scale == "decibels":
        Pxx = 10 * np.log10(Pxx / Pxx_max)
    elif scale != "volts":
        raise ValueError(f"Unsupported scale {scale}")

    return (f, Pxx, Pxx_max)


def plot_psd(ax, f, Pxx, scale="volts"):
    sns.lineplot(x=f, y=Pxx, ax=ax)
    simuran.despine()
    ax.set_xlabel("Frequency (Hz)")

    ylabel = None
    if scale == "volts":
        micro = UnicodeGrabber.get("micro")
        pow2 = UnicodeGrabber.get("pow2")
        ylabel = f"PSD ({micro}V{pow2} / Hz)"
    else:
        ylabel = "PSD (dB)"

    ax.set_ylabel(ylabel)


def per_animal_psd(recording_container, base_dir, figures, **kwargs):
    simuran.set_plot_style()
    item_list = [r.results for r in recording_container]
    parsed_info = []

    scale = kwargs.get("psd_scale", "volts")

    fmt = kwargs.get("image_format", "png")

    for item_dict in item_list:
        item_dict = item_dict["powers"]
        for r in ["SUB", "RSC"]:
            id_ = item_dict[r + " welch"]
            powers = id_[1]
            for v1, v2, v3, v4 in zip(id_[0], powers, id_[2], id_[3]):
                parsed_info.append([v1, v2, v3, v4])

    data = np.array(parsed_info)
    df = pd.DataFrame(data, columns=["frequency", "power", "Group", "region"])
    df[["frequency", "power"]] = df[["frequency", "power"]].apply(pd.to_numeric)

    for ci, oname in zip([95, None], ["_ci", ""]):
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df[df["region"] == "SUB"],
            x="frequency",
            y="power",
            ci=ci,
            estimator=np.median,
            ax=ax,
        )
        simuran.despine()
        plt.xlabel("Frequency (Hz)")

        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            ax.set_ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            ax.set_ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.tight_layout()

        name = recording_container.base_dir[len(base_dir + os.sep) :].replace(
            os.sep, "--"
        )
        out_name = os.path.join("per_animal_psd", name + "--sub--power{}".format(oname))
        fig = simuran.SimuranFigure(fig, out_name, dpi=400, done=True, format=fmt)
        figures.append(fig)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df[df["region"] == "RSC"],
            x="frequency",
            y="power",
            ci=ci,
            estimator=np.median,
            ax=ax,
        )
        simuran.despine()
        plt.xlabel("Frequency (Hz)")
        if scale == "volts":
            micro = UnicodeGrabber.get("micro")
            pow2 = UnicodeGrabber.get("pow2")
            ax.set_ylabel(f"PSD ({micro}V{pow2} / Hz)")
        elif scale == "decibels":
            ax.set_ylabel("PSD (dB)")
        else:
            raise ValueError("Unsupported scale {}".format(scale))
        plt.tight_layout()

        out_name = os.path.join("per_animal_psd", name + "--rsc--power{}".format(oname))
        fig = simuran.SimuranFigure(fig, out_name, dpi=400, done=True, format=fmt)
        figures.append(fig)


def define_recording_group(base_dir):
    dirs = base_dir.split(os.sep)
    if dirs[-1].startswith("CS") or dirs[-2].startswith("CS"):
        group = "Control"
    elif dirs[-1].startswith("LS") or dirs[-2].startswith("LS"):
        group = "Lesion"
    else:
        group = "Undefined"
    return group


def name_plot(recording, base_dir, end):
    return recording.get_name_for_save(base_dir) + end


def powers(
    recording, base_dir, figures, clean_method="avg", fmin=1, fmax=100, **kwargs
):
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    signals_grouped_by_region = lc.clean(
        recording, fmin, fmax, method_kwargs=clean_kwargs
    )["signals"]
    fmt = kwargs.get("image_format", "png")
    psd_scale = kwargs.get("psd_scale", "volts")
    theta_min = kwargs.get("theta_min", 6)
    theta_max = kwargs.get("theta_max", 10)
    delta_min = kwargs.get("delta_min", 1.5)
    delta_max = kwargs.get("delta_max", 4.0)
    plot_psd_ = kwargs.get("plot_psd", True)

    results = {}
    window_sec = 2

    if plot_psd_:
        simuran.set_plot_style()

    for name, signal in signals_grouped_by_region.items():
        results["{} delta".format(name)] = np.nan
        results["{} theta".format(name)] = np.nan
        results["{} low gamma".format(name)] = np.nan
        results["{} high gamma".format(name)] = np.nan
        results["{} total".format(name)] = np.nan

        results["{} delta rel".format(name)] = np.nan
        results["{} theta rel".format(name)] = np.nan
        results["{} low gamma rel".format(name)] = np.nan
        results["{} high gamma rel".format(name)] = np.nan

        sig_in_use = signal.to_neurochat()
        delta_power = sig_in_use.bandpower(
            [delta_min, delta_max], window_sec=window_sec, band_total=True
        )
        theta_power = sig_in_use.bandpower(
            [theta_min, theta_max], window_sec=window_sec, band_total=True
        )
        low_gamma_power = sig_in_use.bandpower(
            [30, 55], window_sec=window_sec, band_total=True
        )
        high_gamma_power = sig_in_use.bandpower(
            [65, 90], window_sec=window_sec, band_total=True
        )

        if not (
            delta_power["total_power"]
            == theta_power["total_power"]
            == low_gamma_power["total_power"]
            == high_gamma_power["total_power"]
        ):
            raise ValueError("Unequal total powers")

        results["{} delta".format(name)] = delta_power["bandpower"]
        results["{} theta".format(name)] = theta_power["bandpower"]
        results["{} low gamma".format(name)] = low_gamma_power["bandpower"]
        results["{} high gamma".format(name)] = high_gamma_power["bandpower"]
        results["{} total".format(name)] = delta_power["total_power"]

        results["{} delta rel".format(name)] = delta_power["relative_power"]
        results["{} theta rel".format(name)] = theta_power["relative_power"]
        results["{} low gamma rel".format(name)] = low_gamma_power["relative_power"]
        results["{} high gamma rel".format(name)] = high_gamma_power["relative_power"]

        # Do power spectra
        sr = signal.sampling_rate
        group = define_recording_group(base_dir)
        r1, r2 = calculate_psd(
            signal, ax, sr, group, name, fmin=fmin, fmax=fmax, scale=psd_scale
        )
        results["{} welch".format(name)] = r1
        results["{} max f".format(name)] = r2

        if plot_psd_:
            fig, ax = plt.subplots()
            plot_psd(ax, r1[0], r1[1], scale=psd_scale)
            out_name = name_plot(recording, base_dir, f"_power_{name}")
            fig = simuran.SimuranFigure(fig, out_name, dpi=400, done=True, format=fmt)
            figures.append(fig)

    return results
