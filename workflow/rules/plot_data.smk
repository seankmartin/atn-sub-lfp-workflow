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
        "results/summary/averaged_psds_psd.csv",
        "results/summary/averaged_signals_psd.csv"
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
        "results/summary/openfield_coherence.csv"
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
        "results/summary/openfield_speed.csv"
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
        "results/summary/openfield_sta.csv",
        "results/summary/openfield_sfc.csv",
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
        "results/summary/muscimol_sta.csv",
        "results/summary/muscimol_sfc.csv",
    output:
        directory("results/plots/spike_lfp_musc/")
    log:
        "logs/plot_spike_lfp_musc.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_spike_lfp.py"

rule plot_tmaze:
    input:
        "results/tmaze/decoding.csv",
        "results/tmaze/results.csv"
    output:
        directory("results/plots/tmaze")
    log:
        "logs/plot_tmaze.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_tmaze.py"

rule plot_sleep:
    input:
        "results/sleep/ripples.pkl",
        "results/sleep/spindles.pkl",
        "results/every_processed_nwb.csv"
    output:
        directory("results/plots/sleep"),
        "results/sleep/spindles2.csv",
        "results/sleep/ripples2.csv",
    log:
        "logs/plot_sleep.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/plot_sleep.py"

    
rule plot_assorted:
    input:
        "results/summary/signal_bandpowers.csv",
    output:
        "results/plots/summary/png/bandpower_SUB.png"