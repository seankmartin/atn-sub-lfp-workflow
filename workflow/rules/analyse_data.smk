rule process_lfp:
    input:
        "results/openfield_nwb.csv"
    output:
        "results/processed_nwbfiles.csv"
    log:
        "logs/process_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_lfp.py"