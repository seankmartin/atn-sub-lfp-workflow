import csv
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from neuronal import LFPDecoder
from skm_pyutils.table import df_from_file


def main(input_dir, config, out_dir, do_coherence, do_decoding):
    coh_df, power_df, res_df, groups, choices, new_lfp = load_saved_results(
        input_dir, config
    )

    if do_coherence:
        plot_coherence_results(res_df, coh_df, power_df, out_dir)
    if do_decoding:
        groups = np.array(groups)
        labels = np.array(choices)
        decoding(new_lfp, groups, labels, out_dir)


def plot_coherence_choice(coherence_df, out_dir):
    coherence_df["Trial result"] = coherence_df["Trial"]
    coherence_df_sub_bit = coherence_df[
        (coherence_df["Part"] == "choice") & (coherence_df["Trial"] != "Forced")
    ]

    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=coherence_df_sub_bit,
        x="Frequency (Hz)",
        y="Coherence",
        hue="Group",
        style="Trial result",
        ci=95,
        estimator=np.median,
    )
    ax.set_ylim(0, 1)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / "choice_coherence_ci")
    fig.save()


def plot_choice_power(power_df, out_dir):
    power_df["Trial result"] = power_df["Trial"]
    power_df_sub_bit = power_df[
        (power_df["Part"] == "choice") & (power_df["Trial"] != "Forced")
    ]
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=power_df_sub_bit,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Group",
        style="Trial result",
        estimator=np.median,
        ci=95,
    )
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / "choice_power_ci")
    fig.save()


def plot_banded_coherence(out_dir, res_df):
    res_df = res_df[res_df["part"] == "choice"]
    plot_bar_coherence(res_df, "Theta", out_dir)
    plot_bar_coherence(res_df, "Delta", out_dir)


def plot_grouped_power_coherence(out_dir, coherence_df, power_df):
    for group in ("Control", "Lesion (ATNx)"):
        coherence_df_sub = coherence_df[coherence_df["Group"] == group]
        power_df_sub = power_df[power_df["Group"] == group]
        plot_group_coherence(group, coherence_df_sub, out_dir)
        plot_group_power(group, power_df_sub, out_dir)


def plot_group_power(group, power_df_sub, out_dir):
    fig, ax = plt.subplots()
    sns.lineplot(
        data=power_df_sub,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Part",
        style="Trial",
        ci=95,
        estimator=np.median,
        ax=ax,
    )
    ax.set_xlim(0, 40)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_power_ci")
    fig.save()


def plot_group_coherence(group, coherence_df_sub, out_dir):
    fig, ax = plt.subplots()
    for ci, ci_name in zip((None, 95), ("", "_ci")):
        sns.lineplot(
            data=coherence_df_sub,
            x="Frequency (Hz)",
            y="Coherence",
            hue="Part",
            style="Trial",
            ci=ci,
            estimator=np.median,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        smr.despine()
        fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_coherence{ci_name}")
        fig.save()


def plot_bar_coherence(res_df, band: str, out_dir):
    fig, ax = plt.subplots()
    sns.barplot(
        data=res_df,
        x="trial",
        y=f"{band}_coherence",
        hue="Group",
        estimator=np.median,
        ax=ax,
    )
    ax.set_xlabel("Trial result")
    ax.set_ylabel(f"{band} coherence")
    plt.tight_layout()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{band} coherence")
    fig.save()


def plot_coherence_results(res_df, coherence_df, power_df, out_dir):
    plot_banded_coherence(out_dir, res_df)
    plot_grouped_power_coherence(out_dir, coherence_df, power_df)
    plot_choice_power(power_df, out_dir)
    plot_coherence_choice(coherence_df, out_dir)


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


def load_saved_results(out_dir, config):
    lfp_len = config["tmaze_lfp_len"]
    decoding_loc = out_dir / "decoding.csv"
    groups, choices, new_lfp = []
    with open(decoding_loc, "r") as f:
        csvreader = csv.reader(f, delimiter=",")
        for row in csvreader:
            groups.append(row[0])
            choices.append(row[1])
            vals = row[2:]
            new_lfp.append(np.array([float(v) for v in vals[:lfp_len]]))

    coh_loc = out_dir / "coherence.csv"
    power_loc = out_dir / "power.csv"
    results_loc = out_dir / "results.csv"
    coherence_df = df_from_file(coh_loc)
    power_df = df_from_file(power_loc)
    res_df = df_from_file(results_loc)

    return coherence_df, power_df, res_df, groups, choices, np.array(new_lfp)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        Path(snakemake.input[0]),
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]),
        snakemake.params["do_coherence"],
        snakemake.params["do_decoding"],
    )
