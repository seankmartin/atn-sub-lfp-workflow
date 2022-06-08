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
