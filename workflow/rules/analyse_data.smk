rule process_lfp:
    input:
        "results/openfield_nwb.csv",
        "results/openfield_cells_nwb.csv",
        "results/muscimol_cells_nwb.csv",
        "results/tmaze_times_nwb.csv",
    output:
        "results/openfield_processed.csv",
        "results/openfield_cells_processed.csv",
        "results/muscimol_cells_processed.csv",
        "results/tmaze_times_processed.csv",
        "results/processed_nwbfiles.csv",
    log:
        "logs/process_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_lfp.py"


rule analyse_tmaze:
    input:
        "results/tmaze_times_processed.csv"
    output:
        "results/tmaze/results.csv",
        "results/tmaze/coherence.csv",
        "results/tmaze/power.csv",
        "results/tmaze/decoding.csv",
    log:
        "logs/process_tmaze.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/analyse_tmaze.py"


rule analyse_spike_lfp:
    input:
        "results/openfield_cells_processed.csv",
        "results/muscimol_cells_processed.csv",
    output:
        "results/summary/openfield_sta.csv",
        "results/summary/openfield_sfc.csv",
        "results/summary/openfield_peak_sfc.csv",
        "results/summary/muscimol_sta.csv",
        "results/summary/muscimol_sfc.csv",
        "results/summary/muscimol_peak_sfc.csv",
    log:
        "logs/analyse_spike_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/analyse_spike_lfp.py"


rule create_dfs:
    input:
        "results/openfield_processed.csv",
        "results/every_processed_nwb.csv"
    output:
        "results/summary/averaged_signals_psd.csv",
        "results/summary/averaged_psds_psd.csv",
        "results/summary/signal_bandpowers.csv",
        "results/summary/openfield_coherence.csv",
        "results/summary/coherence_stats.csv",
        "results/summary/openfield_speed.csv",
    log:
        "logs/create_dfs.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/create_dfs.py"


rule hypothesis_tests:
    input:
        "results/summary/signal_bandpowers.csv",
        "results/summary/coherence_stats.csv",
        "results/summary/speed_theta_avg.csv",
        "results/summary/openfield_peak_sfc.csv",
        "results/summary/musc_spike_lfp_sub_pairs.csv",
        "results/summary/musc_spike_lfp_sub_pairs_later.csv",
        "results/tmaze/results.csv",
        "results/summary/muscimol_peak_sfc.csv",
        "results/sleep/spindles2.csv",
        "results/sleep/ripples2.csv"
    output:
        directory("results/plots/stats"),
        "results/stats_output.txt"
    params:
        show_quartiles=True
    log:
        "logs/stats_tests.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/stats_tests.py"


rule tmaze_decoding:
    input:
        "results/tmaze/decoding.csv"
    output:
        directory("results/plots/tmaze_decoding")
    log:
        "logs/decode_tmaze.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/decode_tmaze.py"


rule ca1_lfp:
    output:
        directory("results/ca1_analysis")
    log:
        "logs/ca1_analysis.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/atnx_ca1_lfp.py"


rule theta_gamma:
    input:
        "results/openfield_processed.csv"
    output:
        directory("results/plots/theta_gamma")
    log:
        "logs/theta_gamma.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/analyse_theta_gamma.py"


rule analyse_sleep:
    input:
        "results/every_processed_nwb.csv",
    output:
        "results/sleep/spindles.pkl",
        "results/sleep/ripples.pkl",
    log:
        "logs/sleep.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/analyse_sleep.py"


rule analyse_abs_power:
    input:
        "results/openfield_processed.csv",
    output:
        "results/summary/bandpowers_abs.csv"
    log:
        "logs/abs_power.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/analyse_absolute_power.py"


rule process_dfs:
    input:
        "results/summary/openfield_speed.csv",
        "results/summary/openfield_peak_sfc.csv",
        "results/summary/muscimol_peak_sfc.csv"
    output:
        "results/summary/speed_theta_avg.csv",
        "results/summary/open_spike_lfp_ns.csv",
        "results/summary/open_spike_lfp_sub.csv",
        "results/summary/musc_spike_lfp_sub.csv",
        "results/summary/musc_spike_lfp_sub_pairs.csv",
        "results/summary/musc_spike_lfp_sub_pairs_later.csv",
    log:
        "logs/process_dfs.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_dfs.py"