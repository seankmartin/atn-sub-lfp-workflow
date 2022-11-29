from pathlib import Path
import numpy as np

from skm_pyutils.table import list_to_df, df_from_file, df_to_file
import simuran as smr


def main(input_df_path, output_dir, config_path):
    input_df = df_from_file(input_df_path)
    cfg = smr.config_from_file(config_path)
    process_speed_theta(input_df, output_dir, cfg)


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
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "summary" / "openfield_speed.csv",
            here / "results" / "summary",
            here / "config" / "simuran_params.yml",
        )
