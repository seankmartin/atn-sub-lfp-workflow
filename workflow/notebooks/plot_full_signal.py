# %%
import sys
from pathlib import Path

import simuran as smr

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))
from plot_signals import plot_all_signals

# %%
loader = smr.loader("nwb")
# path_to_file = r"D:\atn-sub-lfp-workflow\results\processed\LRS1--t_maze--23032018_t2--S3--23032018_LRS1_+maze_t2_3.nwb"
path_to_file = r"D:\atn-sub-lfp-workflow\results\processed\LRS1--t_maze--04042018_t6--S5--04042018_LRS1_+maze_t6_5.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()
nwbfile = recording.data
output_path = Path("test_signals.png")

plot_all_signals(recording, output_path)
