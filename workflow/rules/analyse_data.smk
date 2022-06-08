rule process_openfield_lfp:
    input:
        "results/openfield_nwb.csv"
    output:
        "results/processed_nwbfiles.csv"
    log:
        "logs/process_openfieldlfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_openfield_lfp.py"