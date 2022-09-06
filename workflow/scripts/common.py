import numpy as np
from neurochat.nc_lfp import NLfp


def rename_rat(rat_name):

    rat_name_dict = {
        "CSubRet1": "CSR1",
        "CSubRet2_sham": "CSR2",
        "CSubRet3_sham": "CSR3",
        "CSubRet4": "CSR4",
        "CSubRet5_sham": "CSR5",
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
