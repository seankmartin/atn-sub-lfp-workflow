"""Create some final plots."""
import itertools
from pathlib import Path
import seaborn as sns
from skm_pyutils.table import df_from_file, list_to_df
import matplotlib.pyplot as plt
import simuran as smr


def main(bandpower_df_path, output_dir, cfg_path):
    config = smr.config_from_file(cfg_path)
    bandpower_df = df_from_file(bandpower_df_path)
    smr.set_plot_style()
    plot_bandpower(bandpower_df, output_dir, config)
    pass


def plot_bandpower(input_df, output_dir, config):
    new_list = []
    names = [val for val in input_df.columns if val.endswith("Rel")]
    name_to_centre = {
        "Delta": (config["delta_min"] + config["delta_max"]) / 2,
        "Theta": (config["theta_min"] + config["theta_max"]) / 2,
        "Beta": (config["beta_min"] + config["beta_max"]) / 2,
        "Low Gamma": (config["low_gamma_min"] + config["low_gamma_max"]) / 2,
        "High Gamma": (config["high_gamma_min"] + config["high_gamma_max"]) / 2,
    }
    for i, row in input_df.iterrows():
        for val in names:
            if (val[:3] == "RSC") and (not row["RSC on target"]):
                continue
            centre = name_to_centre[val[4:-4]]
            new_list.append(
                [row[val], centre * row[val], row["Condition"], val[:3], val[4:-4]]
            )
    headers = [
        "Relative Bandpower",
        "Bandpower x Frequency",
        "Condition",
        "Brain Region",
        "Band",
    ]
    new_df = list_to_df(new_list, headers=headers)
    for br, y in itertools.product(
        ["SUB", "RSC"], ["Relative Bandpower", "Bandpower x Frequency"]
    ):
        sub_df = new_df[new_df["Brain Region"] == br]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            data=sub_df,
            y=y,
            x="Band",
            hue="Condition",
            palette="pastel",
            ax=ax,
            showfliers=False,
            width=0.9,
            # palette=["0.65", "r"],
        )
        sns.stripplot(
            data=sub_df,
            y=y,
            x="Band",
            hue="Condition",
            ax=ax,
            # palette="dark:grey",
            palette=["0.4", "0.75"],
            alpha=0.95,
            dodge=True,
            edgecolor="k",
            linewidth=1,
            size=4.5,
            legend=False,
        )
        smr_fig = smr.SimuranFigure(fig, output_dir / f"bandpower_{br}_{y}")
        smr_fig.save()


if __name__ == "__main__":
    try:
        a = snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True

    if use_snakemake:
        main(
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "summary" / "signal_bandpowers.csv",
            here / "results" / "plots" / "summary",
            here / "config" / "simuran_params.yml",
        )
