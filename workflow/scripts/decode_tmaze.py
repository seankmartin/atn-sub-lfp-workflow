import csv
from pathlib import Path
from pprint import pprint

import numpy as np
import simuran as smr

from decoding import LFPDecoder


def main(input_fpath, config, out_dir):
    config = smr.config_from_file(config)
    groups, choices, new_lfp = load_saved_results(input_fpath, config)
    groups = np.array(groups)
    labels = np.array(choices)
    decoding(new_lfp, groups, labels, out_dir)


def load_saved_results(decoding_loc, config):
    lfp_len = config["tmaze_lfp_len"]
    groups, choices, new_lfp = [], [], []
    with open(decoding_loc, "r") as f:
        csvreader = csv.reader(f, delimiter=",")
        for row in csvreader:
            groups.append(row[0])
            choices.append(row[1])
            vals = row[2:]
            new_lfp.append(np.array([float(v) for v in vals[:lfp_len]]))

    return groups, choices, np.array(new_lfp)


def decoding(lfp_array, groups, labels, base_dir):
    smr.set_plot_style()
    for group in ["Control", "Lesion (ATNx)"]:
        correct_groups = groups == group
        lfp_to_use = lfp_array[correct_groups, :]
        labels_ = labels[correct_groups]

        decoder = LFPDecoder(
            labels=labels_,
            mne_epochs=None,
            features=lfp_to_use,
            cv_params={"n_splits": 100},
        )
        out = decoder.decode()
        print(decoder.decoding_accuracy(out[2], out[1]))

        print("\n----------Cross Validation-------------")
        decoder.cross_val_decode(shuffle=False)
        pprint(decoder.cross_val_result)
        pprint(decoder.confidence_interval_estimate("accuracy"))

        print("\n----------Cross Validation Control (shuffled)-------------")
        decoder.cross_val_decode(shuffle=True)
        pprint(decoder.cross_val_result)
        pprint(decoder.confidence_interval_estimate("accuracy"))

        random_search = decoder.hyper_param_search(verbose=True, set_params=False)
        print("Best params:", random_search.best_params_)

        decoder.visualise_features(output_folder=base_dir, name=f"_{group}")


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        Path(snakemake.input[0]),
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]),
    )
