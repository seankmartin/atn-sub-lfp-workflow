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
