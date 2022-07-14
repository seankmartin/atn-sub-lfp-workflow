rule process_lfp:
    input:
        "results/openfield_nwb.csv",
        "results/openfield_cells_nwb.csv",
        "results/muscimol_cells_nwb.csv"
    output:
        "results/openfield_processed.csv",
        "results/openfield_cells_processed.csv",
        "results/muscimol_cells_processed.csv",
        "results/processed_nwbfiles.csv",
    log:
        "logs/process_lfp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/process_lfp.py"


rule process_tmaze:
    input:
        "results/tmaze-times_nwb.csv"
    output:
        directory("results/tmaze")
    log:
        "logs/process_tmaze.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/t_maze_analyse.py"