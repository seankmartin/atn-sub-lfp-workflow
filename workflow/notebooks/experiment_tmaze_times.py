# %%
import sys
from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file

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

# %%
for name in names:
    res_loc = parent.parent / "results" / "tmaze_many" / name
    out_dir = parent.parent / "results" / "plots" / "tmaze_many" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    main(res_loc, out_dir)
