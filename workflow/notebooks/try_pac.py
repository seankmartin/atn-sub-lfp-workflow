# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent / "scripts"))

# %% Grab recording
loader = smr.loader("nwb")
path_to_file = r"E:\Repos\atn-sub-lfp-workflow\results\processed\CSR6--screening_small sq--22022018--22022018_CSR6_screening_small sq_1.nwb"
recording = smr.Recording(loader=loader, source_file=path_to_file)
recording.load()
nwbfile = recording.data
config = smr.config_from_file(parent.parent / "config" / "simuran_params.yml")

# %% Grab data from recording
average_signal = nwbfile.processing["average_lfp"]
sub_signal = average_signal["SUB_avg"].data[:]
rsc_signal = average_signal["RSC_avg"].data[:]

# %% Try out PAC
from pactools import Comodulogram

low_fq_range = np.linspace(6, 12, 30)
estimator = Comodulogram(
    fs=250,
    low_fq_range=low_fq_range,
    low_fq_width=1.0,
    method="duprelatour",
    progress_bar=True,
    random_state=0,
    n_jobs=1,
    n_surrogates=200,
)
fig, ax = plt.subplots(2, 1, figsize=(6, 4))
estimator.fit(sub_signal, rsc_signal)
estimator.plot(
    axs=[ax[0]],
    contour_method="comod_max",
    contour_level=0.05,
)
ax[0].set_title("SUB phase")
estimator.fit(rsc_signal, sub_signal)
estimator.plot(axs=[ax[1]])
ax[1].set_title("RSC phase")

plt.show()

# %%
