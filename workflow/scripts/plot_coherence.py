from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, list_to_df


def create_coherence_df(recording_container):
    l = []
    for recording in recording_container.load_iter():
        nwbfile = recording.data
        coherence_df = nwbfile.processing["lfp_coherence"][
            "coherence_table"
        ].to_dataframe()
        region = coherence_df["label"].values[0]
        group = recording.attrs["treatment"]
        l.extend(
            [group, region, f_val, c_val]
            for f_val, c_val in zip(
                coherence_df["frequency"].values[0], coherence_df["coherence"].values[0]
            )
        )
    headers = ["Group", "Regions", "Frequency (Hz)", "Coherence"]
    return list_to_df(l, headers)


def plot_coherence(df, out_dir, max_frequency=40):
    smr.set_plot_style()

    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 5)", inplace=True)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df[df["Frequency (Hz)"] <= max_frequency],
        x="Frequency (Hz)",
        y="Coherence",
        style="Group",
        hue="Group",
        estimator=np.median,
        ci=95,
        ax=ax,
    )

    plt.ylim(0, 1)
    smr.despine()
    filename = out_dir / "coherence"
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def main(input_df_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)

    coherence_df = create_coherence_df(rc)
    plot_coherence(coherence_df, out_dir, config["max_psd_freq"])


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent.parent,
        snakemake.config["simuran_config"],
    )
