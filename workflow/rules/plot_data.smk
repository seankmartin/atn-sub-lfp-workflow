rule plot_lfp_spectra:
    input:
        "results/processed_nwbfiles.txt"
    output:
        "results/plots.txt"
    log:
        "logs/plot_spectra.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_lfp.py"