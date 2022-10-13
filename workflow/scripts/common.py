import numpy as np
from neurochat.nc_lfp import NLfp


def rename_rat(rat_name):
    if rat_name.endswith("_muscimol"):
        rat_name = rat_name[: -len("_muscimol")]
    if rat_name.endswith("_musc_use"):
        rat_name = rat_name[: -len("_musc_use")]

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
        "CanCSRetCa1": "CanCSRCa1",
        "CanCSRetCa2": "CanCSRCa2",
        "CanCCaRet1": "CanCCaR1",
        "CanCCaRet2": "CanCCaR2",
        "CanCsubRet8": "CanCSR8",
        "CanCSubCaR2": "CanCSCaR2",
    }

    return rat_name_dict.get(rat_name, rat_name)


def animal_to_mapping(s):
    cl_13 = "CL-SR_1-3.py"
    cl_46 = "CL-SR_4-6.py"
    d = {
        "CSR1": cl_13,
        "CSR2_sham": cl_13,
        "CSR3_sham": cl_13,
        "CSR4": cl_46,
        "CSR5_sham": cl_46,
        "CSR6": cl_46,
        "LSR1": cl_13,
        "LSR2": cl_13,
        "LSR3": cl_13,
        "LSR4": cl_46,
        "LSR5": cl_46,
        "LSR6": cl_46,
        "LSR7": "LSR7.py",
        "LRS1": "CL-RS.py",
        "CRS1": "CL-RS.py",
        "CRS2": "CL-RS.py",
        "CanCSCa1": "CanCSCa.py",
        "CanCSR7": "CanCSR.py",
        "CanCSR8": "CanCSR.py",
        "CanCSRCa1": "CanCSRCa.py",
        "CanCSRCa2": "CanCSRCa.py",
        "CanCCaR1": "CanCCaR.py",
        "CanCCaR2": "CanCCaR.py",
        "CanCSCaR1": "CanCSCaR.py",
        "CanCSCaR2": "CanCSCaR.py",
        "CanCSCaR4": "CanCSCaR.py",
        "CanCSCaR5": "CanCSCaR.py",
    }

    return d.get(s, "no_mapping")


def filename_to_mapping(s):
    """Some filenames need special mappings."""
    d = {
        "16082017_CSubRet1_smallsq_1.set": "only_1_sub_eeg.py",
        "23112017_LSubRet5_smallsq_screen_6.set": "only_1_sub_eeg.py",
        "26112017_LSubRet5_smallsq_screen_7.set": "only_1_sub_eeg.py",
    }

    return d.get(s, np.nan)


def rsc_histology(rat_name):

    rat_name_dict = {
        "CSR1": "midline",
        "CSR2_sham": "midline",
        "CSR3_sham": "contralateral",
        "CSR4": "midline",
        "CSR5_sham": "ipsilateral",
        "CSR6": "ipsilateral",
        "LSR1": "ipsilateral",
        "LSR2": "contralateral",
        "LSR3": "contralateral",
        "LSR4": "ipsilateral",
        "LSR5": "contralateral",
        "LSR6": "contralateral",
        "LSR7": "ipsilateral",
        "LRS1": "ipsilateral",
        "CRS1": "ipsilateral",
        "CRS2": "ipsilateral",
        "CanCSCa1": "NA",
        "CanCSR7": "contralateral",
        "CanCSR8": "ipsilateral",
        "CanCSRCa1": "contralateral",
        "CanCSRCa2": "contralateral",
        "CanCCaR1": "not imaged",
        "CanCCaR2": "ipsilateral",
        "CanCSCaR1": "not imaged",
        "CanCSCaR2": "ipsilateral",
        "CanCSCaR4": "not imaged",
        "CanCSCaR5": "ipsilateral",
    }

    return rat_name_dict[rat_name]


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
