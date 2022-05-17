rule test:
    output:
        "results/blah.py"
    input:
        "results/axona_file_index.csv"
    shell:
        r"python E:\Repos\SIMURAN\simuran\main\main_from_template.py results\axona_file_index.csv lfp_atn_simuran\configs\default.py lfp_atn_simuran\functions\fn_spectra.py --data-filterpath lfp_atn_simuran\table_params\CSR1.yaml"