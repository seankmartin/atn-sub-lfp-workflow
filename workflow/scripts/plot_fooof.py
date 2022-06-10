from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from fooof import FOOOFGroup
from skm_pyutils.table import df_from_file, list_to_df

smr.set_plot_style()


def grab_fooof_info_from_container(recording_container):
    info_for_fooof = {"Control": {}, "Lesion": {}}
    for recording in recording_container.load_iter():
        brain_regions = recording.data.electrodes.to_dataframe()["location"]
        brain_regions = sorted(list(set(brain_regions)))
        group = recording.attrs["treatment"].capitalize()
        psd_table = recording.data.processing["lfp_power"][
            "power_spectra"
        ].to_dataframe()

        for region in brain_regions:
            psd_row = psd_table.loc[psd_table["label"] == f"{region}_avg"]
            max_psd = psd_row["max_psd"].values[0]
            power = np.array(psd_row["power"].values[0]).astype(np.float64)
            volts_scale = np.power(10.0, (power / 10.0)) * max_psd
            if region not in info_for_fooof[group]:
                info_for_fooof[group][region] = {"frequency": None, "spectra": []}
            info_for_fooof[group][region]["spectra"].append(volts_scale)
            info_for_fooof[group][region]["frequency"] = psd_row["frequency"].values[0]

    return info_for_fooof


def plot_all_fooof(info_for_fooof, out_dir, fmax=40):
    peaks_data = []
    for group in sorted(info_for_fooof.keys()):
        for region in sorted(info_for_fooof[group].keys()):
            fg = FOOOFGroup(
                peak_width_limits=[1.0, 8.0],
                max_n_peaks=2,
                min_peak_height=0.1,
                peak_threshold=2.0,
                aperiodic_mode="fixed",
            )

            fooof_arr_s = np.array(info_for_fooof[group][region]["spectra"])
            fooof_arr_f = np.array(info_for_fooof[group][region]["frequency"])
            fg.fit(fooof_arr_f, fooof_arr_s, [0.5, fmax], progress="tqdm")
            out_name = f"{region}--{group}--fooof.pdf"
            out_dir.mkdir(parents=True, exist_ok=True)
            fg.save_report(out_name, out_dir)

            peaks = fg.get_params("peak_params", 0)[:, 0]
            peaks_data.extend([p, group, region] for p in peaks)
    return peaks_data


def plot_fooof_peaks(peaks_data, out_dir):
    peaks_df = list_to_df(peaks_data, headers=["Peak frequency", "Group", "Region"])

    for r in sorted(list(set(peaks_df["Region"]))):
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
        smr.despine()
        ax.set_title(f"{r} Peak frequencies (Hz)")
        ax.set_xlim(2, 20)
        out_name = out_dir / f"{r}--fooof_combined"
        smr_fig = smr.SimuranFigure(fig, out_name)
        smr_fig.save()


def main(input_df_path, output_directory, config_path):
    loader = smr.loader("nwb")
    cfg = smr.ParamHandler(source_file=config_path)
    rc = smr.RecordingContainer.from_table(df_from_file(input_df_path), loader)

    fooof_info = grab_fooof_info_from_container(rc)
    peaks_data = plot_all_fooof(fooof_info, output_directory, cfg["max_psd_freq"])
    plot_fooof_peaks(peaks_data, output_directory)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
