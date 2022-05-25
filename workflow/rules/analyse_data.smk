rule process_openfield_lfp:
    input:
        "results/openfield_nwb.csv"
    output:
        "results/sim_results--subret_recordings.csv"
    log:
        "logs/process_openfieldlfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_openfield_lfp.py"