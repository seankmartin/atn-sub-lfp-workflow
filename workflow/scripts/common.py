import numpy as np
from neurochat.nc_lfp import NLfp


def rename_rat(rat_name):

    rat_name_dict = {
        "CSubRet1": "CSR1",
        "CSubRet2_sham": "CSR2_sham",
        "CSubRet3_sham": "CSR3_sham",
        "CSubRet4": "CSR4",
        "CSubRet5_sham": "CSR5_sham",
        "CSR6": "CSR6",
        "LSubRet1": "LSR1",
        "LSubRet2": "LSR2",
        "LSubRet3": "LSR3",
        "LSubRet4": "LSR4",
        "LSubRet5": "LSR5",
        "LSR6": "LSR6",
        "LSR7": "LSR7",
    }

    return rat_name_dict.get(rat_name, rat_name)


def rsc_histology(rat_name):

    rat_name_dict = {
        "CSubRet1": "midline",
        "CSubRet2_sham": "midline",
        "CSubRet3_sham": "contralateral",
        "CSubRet4": "midline",
        "CSubRet5_sham": "ipsilateral",
        "CSR6": "ipsilateral",
        "LSubRet1": "ipsilateral",
        "LSubRet2": "contralateral",
        "LSubRet3": "contralateral",
        "LSubRet4": "ipsilateral",
        "LSubRet5": "contralateral",
        "LSR6": "contralateral",
        "LSR7": "ipsilateral",
        "LRS1": "ipsilateral",
        "CRS1": "ipsilateral",
        "CRS2": "ipsilateral",
        "CanCSCa1": "NA",
        "CanCSR7": "contralateral",
        "CanCSR8": "ipsilateral",
        "CanCSRetCa1": "contralateral",
        "CanCSRetCa2": "contralateral",
        "CanCCaRet1": "not imaged",
        "CanCCaRet2": "ipsilateral",
        "CanCSCaR1": "not imaged",
        "CanCSCaR2": "ipsilateral",
        "CanCSCaR4": "not imaged",
        "CanCSCaR5": "ipsilateral",
    }

    return rat_name_dict.get(rat_name, "RSC_not_recorded")


def numpy_to_nc(data, sample_rate=None, timestamp=None):
    if timestamp is None and sample_rate is None:
        raise ValueError("Must provide either sample_rate or timestamp")
    if timestamp is None:
        timestamp = np.arange(0, len(data), dtype=np.float32) / sample_rate
    elif sample_rate is None:
        sample_rate = 1 / np.mean(np.diff(timestamp))

    lfp = NLfp()
    lfp._set_samples(data)
    lfp._set_sampling_rate(sample_rate)
    lfp._set_timestamp(timestamp)
    lfp._set_total_samples(len(data))
    return lfp
