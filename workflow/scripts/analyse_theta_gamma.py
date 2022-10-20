from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import simuran as smr
from pactools import Comodulogram
from skm_pyutils.table import df_from_file


def main(input_, output_dir, config_path):
    # config = smr.config_from_file(config_path)
    datatable = df_from_file(input_)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)

    for i in range(len(rc)):
        if not rc[i].attrs["RSC on target"]:
            continue
        r = rc.load(i)
        nwbfile = r.data
        average_signal = nwbfile.processing["average_lfp"]
        sub_signal = average_signal["SUB_avg"].data[:]
        rsc_signal = average_signal["RSC_avg"].data[:]
        name = r.get_name_for_save()
        out_path = output_dir / f"{name}.png"
        compute_pac(sub_signal, rsc_signal, out_path, 250)


def compute_pac(sub_signal, rsc_signal, out_path, fs=250):
    low_fq_range = np.linspace(6, 12, 30)
    estimator = Comodulogram(
        fs=fs,
        low_fq_range=low_fq_range,
        low_fq_width=1.0,
        method="duprelatour",
        progress_bar=True,
        random_state=0,
        n_jobs=4,
        n_surrogates=150,
    )
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    estimator.fit(sub_signal, rsc_signal)
    estimator.plot(
        axs=[ax[0]],
        contour_method="comod_max",
        contour_level=0.05,
    )
    ax[0].set_title("SUB Phase, RSC amplitude")
    estimator.fit(rsc_signal, sub_signal)
    estimator.plot(axs=[ax[1]])
    ax[1].set_title("RSC phase, SUB amplitude")

    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    try:
        smr.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    except Exception:
        here = Path(__file__).parent.parent
        input_path = here / "results" / "openfield_processed.csv"
        output_dir = here / "results" / "plots" / "theta_gamma"
        config_path = here.parent / "config" / "simuran_params.yml"
        main(input_path, output_dir, config_path)
