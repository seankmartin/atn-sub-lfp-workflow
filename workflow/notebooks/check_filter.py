# %%
from pathlib import Path

import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, filter_table

# %%
here = Path(__file__).resolve().parent
file_path = here.parent.parent / "results" / "axona_file_index.csv"
output_path = file_path.with_name("test.csv")
filter_path = here.parent.parent / "config" / "tmaze_recordings.yml"

# %%
df = df_from_file(file_path)
cfg = smr.ParamHandler(source_file=filter_path)
filtered_df = filter_table(df, cfg)
df_to_file(filtered_df, output_path)

# %%
