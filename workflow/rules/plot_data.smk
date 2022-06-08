rule plot_lfp_spectra:
    input:
        "results/processed_nwbfiles.csv"
    output:
        "results/spectra_plots.txt"
    log:
        "logs/plot_spectra.log"
    params:
        output_dir="results/plots/spectra"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spectra.py"