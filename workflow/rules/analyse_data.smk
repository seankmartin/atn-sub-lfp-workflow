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
