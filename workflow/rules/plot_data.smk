GROUPS = ["Control", "Lesion"]
REGIONS = ["SUB", "RSC"]

rule plot_lfp_spectra:
    input:
        "results/openfield_processed.csv"
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
        "results/openfield_processed.csv"
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
        "results/openfield_processed.csv"
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

rule plot_coherence:
    input:
        "results/openfield_processed.csv"
    output:
        report("results/plots/summary/png/coherence.png", category="Summary")
    log:
        "logs/plot_coherence.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_coherence.py"

rule plot_speed_lfp:
    input:
        "results/openfield_processed.csv"
    output:
        report(
            expand(
                "results/plots/summary/png/{region}--speed_theta.png", region=REGIONS),
            category="Summary")
    log:
        "logs/plot_speed_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_speed_vs_lfp.py"

rule plot_lfp:
    input:
        "results/openfield_processed.csv"
    output:
        directory("results/plots/signals/")
    log:
        "logs/plot_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_signals.py"

rule plot_open_spike_lfp:
    input:
        "results/openfield_processed.csv",
        "workflow/sheets/openfield_cells.csv"
    output:
        directory("results/plots/spike_lfp/")
    log:
        "logs/plot_spike_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spike_lfp.py"

rule plot_musc_spike_lfp:
    input:
        "results/muscimol_cells_processed.csv",
        "workflow/sheets/openfield_cells.csv"
    output:
        directory("results/plots/spike_lfp/")
    log:
        "logs/plot_spike_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spike_lfp.py"