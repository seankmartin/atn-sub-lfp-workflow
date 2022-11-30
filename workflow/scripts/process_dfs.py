from pathlib import Path
import numpy as np
import pandas as pd

from skm_pyutils.table import list_to_df, df_from_file, df_to_file
import simuran as smr


def main(input_df_path, open_spike_path, musc_spike_path, output_dir, config_path):
    input_df = df_from_file(input_df_path)
    cfg = smr.config_from_file(config_path)
    # process_speed_theta(input_df, output_dir, cfg)
    # process_open_spike_lfp(open_spike_path, output_dir, cfg)
    process_musc_spike_lfp(musc_spike_path, output_dir, cfg)


def process_open_spike_lfp(spike_lfp_path, output_dir, config):
    df = df_from_file(spike_lfp_path)

    data_of_interest = (df["Region"] == "SUB") & (df["Spatial"] == "Non-Spatial")
    df_to_file(df[data_of_interest], output_dir / "open_spike_lfp_ns.csv")

    data_of_interest = (df["Region"] == "SUB") & (df["Group"] == "Control")
    df_to_file(df[data_of_interest], output_dir / "open_spike_lfp_sub.csv")


def process_musc_spike_lfp(spike_lfp_path, output_dir, config):
    df = df_from_file(spike_lfp_path)

    data_of_interest = df["Region"] == "SUB"

    # Spike matching
    num_spike_rows = [
        (5, 1),
        (6, 4),
        (2, 3),
        (1, 2),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 2),
        (1, 3),
        (2, 2),
        (1, 3),
        (2, 1),
    ]

    idx_start = []
    current_val = 0
    for val in num_spike_rows:
        idx_start.extend([current_val + (x * val[1]) for x in range(val[0])])
        current_val = idx_start[-1] + val[1]

    units = [(idx_start[0], idx_start[1])]
    units.extend((idx_start[6] + i, idx_start[7] + i) for i in range(4))
    units.extend((idx_start[11] + i, idx_start[13] + i) for i in range(2))
    units.append((idx_start[18] + 1, idx_start[20] + 1))
    units.append((idx_start[19], idx_start[20]))
    units.append((idx_start[19] + 1, idx_start[20] + 2))

    all_dfs = []
    for unit in units:
        row1 = df[data_of_interest].iloc[unit[0]]
        row2 = df[data_of_interest].iloc[unit[1]]
        merged = pd.concat([row1, row2], axis=0).to_frame().T
        all_dfs.append(merged)
    final = pd.concat(all_dfs)
    final.columns = [*df.columns, *[x + "_musc" for x in df.columns]]
    df_to_file(df[data_of_interest], output_dir / "musc_spike_lfp_sub.csv")
    df_to_file(final, output_dir / "musc_spike_lfp_sub_pairs.csv")


def process_speed_theta(input_df, output_dir, config):
    d = {"RSC_Control": {}, "RSC_Lesion": {}, "SUB_Control": {}, "SUB_Lesion": {}}
    final_res = []
    current_id = None
    for i, row in input_df.iterrows():
        power = row["power"]
        speed = row["speed"]
        id_ = row["ID"]
        region = row["region"]
        group = row["Condition"]
        rsc_on_target = row["RSC on target"]
        if id_ != current_id:
            d = add_mean_speeds(config, d, final_res, rsc_on_target)
            current_id = id_

        d[f"{region}_{group}"].setdefault(speed, [])
        d[f"{region}_{group}"][speed].append(power)
    add_mean_speeds(config, d, final_res, rsc_on_target)

    headers = ["power", "speed", "region", "Condition", "RSC on target"]
    df = list_to_df(final_res, headers=headers)
    df_to_file(df, output_dir / "speed_theta_avg.csv")
    df_to_file(
        df[(df["Condition"] == "Control") & (df["region"] == "SUB")],
        output_dir / "speed_theta_avg_ctrl_sub.csv",
    )
    df_to_file(
        df[
            (df["Condition"] == "Control")
            & (df["region"] == "RSC")
            & (df["RSC on target"])
        ],
        output_dir / "speed_theta_avg_ctrl_rsc.csv",
    )
    df_to_file(
        df[(df["Condition"] == "Lesion") & (df["region"] == "SUB")],
        output_dir / "speed_theta_avg_lesion_sub.csv",
    )
    df_to_file(
        df[
            (df["Condition"] == "Lesion")
            & (df["region"] == "RSC")
            & df["RSC on target"]
        ],
        output_dir / "speed_theta_avg_lesion_rsc.csv",
    )


def add_mean_speeds(config, d, final_res, rsc_on_target):
    inter_res = []
    for k, v in d.items():
        r, g = k.split("_")
        for k2, v2 in v.items():
            if k2 > config["max_speed"]:
                continue
            if len(v2) == 0:
                m = np.nan
            else:
                m = np.mean(v2)
            inter_res.append([m, k2, r, g, rsc_on_target])
    for val in sorted(inter_res, key=lambda x: x[1]):
        final_res.append(val)
    d = {
        "RSC_Control": {},
        "RSC_Lesion": {},
        "SUB_Control": {},
        "SUB_Lesion": {},
    }

    return d


if __name__ == "__main__":
    try:
        a = snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        main(
            *snakemake.input,
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "summary" / "openfield_speed.csv",
            here / "results" / "summary" / "openfield_peak_sfc.csv",
            here / "results" / "summary" / "muscimol_peak_sfc.csv",
            here / "results" / "summary",
            here / "config" / "simuran_params.yml",
        )
