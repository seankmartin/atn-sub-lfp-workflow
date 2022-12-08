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
    {"choice": [1.0, 1.0], "start": 2.0, "end": 2.0},
    {"choice": [1.5, 0.5], "start": 2.0, "end": 2.0},
    {"choice": [3.5, 0.5], "start": 4.0, "end": 4.0},
    {"choice": [2.0, 2.0], "start": 4.0, "end": 4.0},
    {"choice": [1.5, 1.5], "start": 3.0, "end": 3.0},
]

names = ["tmaze_2s", "tmaze_2s_uneven", "tmaze_4s", "tmaze_4s_mid", "tmaze_3s"]

# %%
for t_dict, name in zip(ttimes, names):
    out_dir = parent.parent / "results" / "tmaze_many" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config["max_lfp_lengths"] = t_dict
    compute_and_save_coherence(out_dir, config, rc)

# # %%
for name in names:
    res_loc = parent.parent / "results" / "tmaze_many" / name
    out_dir = parent.parent / "results" / "plots" / "tmaze_many" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    main(res_loc, out_dir)

# %%
parts = ["start", "choice", "end"]
trials = ["Forced", "Correct", "Incorrect"]
groups = ["Control", "Lesion (ATNx)"]
full_list = []
for name in names:
    df = df_from_file(parent.parent / "results" / "tmaze_many" / name / "results.csv")
    for g, t, p in itertools.product(groups, trials, parts):
        df_bit = (df["trial"] == t) & (df["part"] == p) & (df["Group"] == g)
        theta_median = df[df_bit]["Theta Coherence"].median()
        beta_median = df[df_bit]["Beta Coherence"].median()
        full_list.append([name, f"{g}_{t}_{p}", theta_median, beta_median])

headers = ["Times", "Grouping", "Theta Coherence", "Beta Coherence"]
df = list_to_df(full_list, headers=headers)
df_to_file(df, parent.parent / "results" / "tmaze_many" / "summary.csv")
