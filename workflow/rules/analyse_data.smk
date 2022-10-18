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
        "../scripts/t_maze_analyse.py"


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
        "results/summary/openfield_speed.csv",
        "results/summary/openfield_peak_sfc.csv",
        "results/tmaze/results.csv",
        "results/summary/muscimol_peak_sfc.csv",
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
        "../scripts/t_maze_decode.py"


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
        "results/other_process.csv",
    output:
        "results/sleep/spindles.pkl",
        "results/sleep/ripples.pkl"
    log:
        "logs/sleep.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/sleep_analysis.py"