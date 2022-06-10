GROUPS = ["Control", "Lesion"]
REGIONS = ["SUB", "RSC"]

rule plot_lfp_spectra:
    input:
        "results/processed_nwbfiles.csv"
    output:
        report(
            directory("results/plots/spectra"),
            patterns=["{name}.png"],
            category="Spectra",
            caption="../report/per_animal.rst")
    log:
        "logs/plot_spectra.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spectra.py"

rule plot_spectra_summary:
    input:
        "results/processed_nwbfiles.csv"
    output:
        report("results/plots/summary/png/per_animal_psds--averaged_psds--SUB.png", category="Comparison"),
        report("results/plots/summary/png/per_animal_psds--averaged_psds--RSC.png", category="Comparison"),
        report("results/plots/summary/png/per_group_psds--averaged_psds--RSC.png", category="Comparison"),
        report("results/plots/summary/png/per_group_psds--averaged_psds--SUB.png", category="Comparison"),
        report("results/plots/summary/png/per_animal_psds--averaged_signals--SUB.png", category="Summary"),
        report("results/plots/summary/png/per_animal_psds--averaged_signals--RSC.png", category="Summary"),
        report("results/plots/summary/png/per_group_psds--averaged_signals--RSC.png", category="Summary"),
        report("results/plots/summary/png/per_group_psds--averaged_signals--SUB.png", category="Summary")
    log:
        "logs/plot_spectra_summary.log"
    params:
        mode="summary"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spectra.py"

rule plot_fooof:
    input:
        "results/processed_nwbfiles.csv"
    output:
        expand("results/plots/summary/{region}--{group}--fooof.pdf", region=REGIONS, group=GROUPS),
        report(
            expand("results/plots/summary/png/{region}--fooof_combined.png", region=REGIONS), category="Summary")
    log:
        "logs/plot_fooof.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_fooof.py"
    