"""Create some final plots."""
from pathlib import Path
import seaborn as sns
from skm_pyutils.table import df_from_file, list_to_df
import matplotlib.pyplot as plt
import simuran as smr


def main(bandpower_df_path, output_dir):
    bandpower_df = df_from_file(bandpower_df_path)
    smr.set_plot_style()
    plot_bandpower(bandpower_df, output_dir)
    pass


def plot_bandpower(input_df, output_dir):
    new_list = []
    names = [val for val in input_df.columns if val.endswith("Rel")]
    for i, row in input_df.iterrows():
        for val in names:
            if (val[:3] == "RSC") and (not row["RSC on target"]):
                continue
            new_list.append([row[val], row["Condition"], val[:3], val[3:-4]])
    headers = ["Relative Bandpower", "Condition", "Brain Region", "Band"]
    new_df = list_to_df(new_list, headers=headers)
    for br in ["SUB", "RSC"]:
        sub_df = new_df[new_df["Brain Region"] == br]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(
            data=sub_df,
            y="Relative Bandpower",
            x="Band",
            hue="Condition",
            palette="pastel",
            ax=ax,
            showfliers=False,
            width=0.9,
        )
        sns.stripplot(
            data=sub_df,
            y="Relative Bandpower",
            x="Band",
            hue="Condition",
            ax=ax,
            palette="dark:grey",
            alpha=0.95,
            dodge=True,
            s=6,
        )
        smr_fig = smr.SimuranFigure(fig, output_dir / f"bandpower_{br}")
        smr_fig.save()


if __name__ == "__main__":
    try:
        a = snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True

    if use_snakemake:
        main(snakemake.input[0], Path(snakemake.output[0]).parent)
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "summary" / "signal_bandpowers.csv",
            here / "results" / "plots" / "summary",
        )
