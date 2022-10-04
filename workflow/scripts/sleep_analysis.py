import pandas as pd

from sleep_utils import mark_rest


def main(input_path, out_dir):
    df = pd.read_csv(input_path, parse_dates=["date_time"])
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    plot_recordings_per_animal(sleep, out_dir / "rat_stats.png")
