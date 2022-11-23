from pathlib import Path

import pandas as pd
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file
from simuran.bridges.neurochat_bridge import signal_to_neurochat


def main(input_, output_dir, config_path):
    config = smr.config_from_file(config_path)
    datatable = df_from_file(input_)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    absolute_power(rc, output_dir, config)


def absolute_power(rc, out_dir, kwargs):
    theta_min = kwargs.get("theta_min", 6)
    theta_max = kwargs.get("theta_max", 10)
    delta_min = kwargs.get("delta_min", 1.5)
    delta_max = kwargs.get("delta_max", 4.0)
    beta_min = kwargs.get("beta_min")
    beta_max = kwargs.get("beta_max")
    low_gamma_min = kwargs.get("low_gamma_min", 30)
    low_gamma_max = kwargs.get("low_gamma_max", 55)
    high_gamma_min = kwargs.get("high_gamma_min", 65)
    high_gamma_max = kwargs.get("high_gamma_max", 90)
    window_sec = 2

    final_dict = {
        "Filename": [],
        "SUB Delta": [],
        "SUB Theta": [],
        "SUB Beta": [],
        "SUB Low Gamma": [],
        "SUB High Gamma": [],
        "RSC Delta": [],
        "RSC Theta": [],
        "RSC Beta": [],
        "RSC Low Gamma": [],
        "RSC High Gamma": [],
        "RSC on target": [],
        "Condition": [],
    }
    for r in rc.load_iter():
        nwbfile = r.data
        lfp = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"]
        electrodes = nwbfile.electrodes.to_dataframe()
        region_powers = {}
        for i, lfp in enumerate(lfp.data[:].T):
            region = electrodes["location"][i]
            if region in region_powers and len(region_powers[region]) > 2:
                continue
            avg_sig = smr.Eeg.from_numpy(lfp, 250)
            sig_in_use = signal_to_neurochat(avg_sig)
            delta_power = sig_in_use.bandpower(
                [delta_min, delta_max], window_sec=window_sec, band_total=True
            )["bandpower"]
            theta_power = sig_in_use.bandpower(
                [theta_min, theta_max], window_sec=window_sec, band_total=True
            )["bandpower"]
            beta_power = sig_in_use.bandpower(
                [beta_min, beta_max], window_sec=window_sec, band_total=True
            )["bandpower"]
            low_gamma_power = sig_in_use.bandpower(
                [low_gamma_min, low_gamma_max], window_sec=window_sec, band_total=True
            )["bandpower"]
            high_gamma_power = sig_in_use.bandpower(
                [high_gamma_min, high_gamma_max], window_sec=window_sec, band_total=True
            )["bandpower"]
            region_powers.setdefault(region, [])
            region_powers[region].append(
                [
                    delta_power,
                    theta_power,
                    beta_power,
                    low_gamma_power,
                    high_gamma_power,
                ]
            )
        avg_dict = {}
        for k, v in region_powers.items():
            avg_dict[k] = [(v[0][i] + v[1][i]) / 2 for i in range(len(v[0]))]
        final_dict["Filename"].append(r.source_file)
        final_dict["RSC on target"].append(r.attrs["RSC on target"])
        final_dict["Condition"].append(r.attrs["treatment"])
        for k, v in avg_dict.items():
            final_dict[f"{k} Delta"].append(v[0])
            final_dict[f"{k} Theta"].append(v[1])
            final_dict[f"{k} Beta"].append(v[2])
            final_dict[f"{k} Low Gamma"].append(v[3])
            final_dict[f"{k} High Gamma"].append(v[4])

    df = pd.DataFrame(final_dict)
    df_to_file(df, out_dir / "bandpowers_abs.csv")


if __name__ == "__main__":
    try:
        a = snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True

    if use_snakemake:
        simuran.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "openfield_processed.csv",
            here / "results" / "summary",
            here / "config" / "simuran_params.yml",
        )
