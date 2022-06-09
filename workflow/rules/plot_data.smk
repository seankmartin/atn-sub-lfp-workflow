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

rule plot_spectra_summary:
    input:
        "results/processed_nwbfiles.csv"
    output:
        report("results/plots/summary/png/per_animal_psds--SUB.png", category="Spectra"),
        report("results/plots/summary/png/per_animal_psds--RSC.png", category="Spectra"),
        report("results/plots/summary/png/per_group_psds--RSC.png", category="Spectra"),
        report("results/plots/summary/png/per_group_psds--SUB.png", caption="spectra_1.rst", category="Spectra")
    log:
        "logs/plot_spectra_summary.log"
    params:
        output_dir="results/plots/summary",
        mode="summary"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spectra.py"