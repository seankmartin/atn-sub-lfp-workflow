# %%
import sys
from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file, list_to_df, df_to_file
import itertools

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

from analyse_tmaze import compute_and_save_coherence
from plot_tmaze import main

# %%
config_path = parent.parent / "config" / "simuran_params.yml"
tmaze_times_filepath = parent.parent / "results" / "tmaze_times_processed.csv"
config = smr.config_from_file(config_path)
df = df_from_file(tmaze_times_filepath)
rc = smr.RecordingContainer.from_table(df, smr.loader("nwb"))

# %%

ttimes = [
    {"choice": [0.9, 0.1], "start": 1.0, "end": 1.0},
    {"choice": [0.8, 0.2], "start": 1.0, "end": 1.0},
    {"choice": [1.3, 0.2], "start": 1.5, "end": 1.5},
]

names = ["tmaze_1s", "tmaze_1s_diff", "tmaze_1-5s"]

# %%
parts = ["start", "choice", "end"]
trials = ["Forced", "Correct", "Incorrect"]
groups = ["Control", "Lesion (ATNx)"]

lfp_name = "tmaze_many_eeg"
tmaze_val = True
full_list = []
config["tmaze_egf"] = tmaze_val
for t_dict, name in zip(ttimes, names):
    out_dir = parent.parent / "results" / lfp_name / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config["max_lfp_lengths"] = t_dict
    compute_and_save_coherence(out_dir, config, rc)

    res_loc = parent.parent / "results" / lfp_name / name
    out_dir = parent.parent / "results" / "plots" / lfp_name / name
    out_dir.mkdir(parents=True, exist_ok=True)
    main(res_loc, out_dir)

    df = df_from_file(parent.parent / "results" / lfp_name / name / "results.csv")
    for g, t, p in itertools.product(groups, trials, parts):
        df_bit = (df["trial"] == t) & (df["part"] == p) & (df["Group"] == g)
        theta_median = df[df_bit]["Theta Coherence"].median()
        beta_median = df[df_bit]["Theta Coherence"].mean()
        full_list.append([name, f"{g}_{t}_{p}", theta_median, beta_median])

headers = ["Times", "Grouping", "Theta Coherence", "Theta mean Coherence"]
df = list_to_df(full_list, headers=headers)
df_to_file(df, parent.parent / "results" / lfp_name / "summary.csv")
