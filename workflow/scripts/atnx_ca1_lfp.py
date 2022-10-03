import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran
from fooof import FOOOFGroup
from lfp_atn_simuran.Scripts.frequency_analysis import powers


def main(input_dir, out_dir, cfg_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = simuran.config_from_file(cfg_path)

    rc = load_rc(input_dir)
    ah = simuran.AnalysisHandler(handle_errors=True)
    sm_figures = []
    fn_kwargs = config

    for r in rc:
        for i in range(len(r.signals)):
            r.signals[i].load()
        fn_args = [r, input_dir, sm_figures]
        ah.add_fn(powers, *fn_args, **fn_kwargs)
    ah.run_all_fns(pbar=True)
    for f in sm_figures:
        f.savefig(filename=out_dir / f.filename)
    ah.save_results_to_table(os.path.join(out_dir, "results.csv"))
    visualise(out_dir, config)


def load_rc(path_dir):
    from simuran.loaders.neurochat_loader import NCLoader

    loader = NCLoader(system="Axona", pos_extension=".pos")
    files_df = loader.index_files(path_dir)
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
    with open(os.path.join(out_dir, "results.csv"), "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",", quotechar="'")
        welch_freqs = []
        welch_powers = []
        freq_end = 199
        power_end = 2 * freq_end
        max_pxxs = []
        for row in csvreader:
            if row[0] == "CA1 welch":
                values = row[1:]
                freqs = values[:freq_end]
                freqs = np.array([float(f[1:]) for f in freqs])
                powers = values[freq_end:power_end]
                powers = np.array([float(f[1:]) for f in powers])
                welch_freqs.append(freqs)
                welch_powers.append(powers)
            elif row[0] == "CA1 max f":
                val = float(row[1])
                max_pxxs.append(val)
    in_list = [np.array(welch_freqs).flatten(), np.array(welch_powers).flatten()]

    # Then combine these into a pandas df as
    # F, P
    df = pd.DataFrame(in_list).T
    df.columns = ["frequency", "power"]
    df.to_csv(os.path.join(out_dir, "power_results.csv"), index=False)
    print(df.head())

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
        plt.xlim(0, config["fmax_plot"])
        plt.ylim(-25, 0)
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
        plt.show()
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
    fg.fit(fooof_arr_f[0], fooof_arr_s, [0.5, 40], progress="tqdm.notebook")

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
    simuran.set_only_log_to_file(snakemake.log[0])
    main(snakemake.config["ca1_directory"], snakemake.output[0])
