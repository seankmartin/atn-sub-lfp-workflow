rule preprocess_data:
    output:
        "results/axona_file_index.csv"
    log:
        "logs/process_data.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/index_axona_files.py"


rule add_data_types:
    input:
        "results/axona_file_index.csv"
    output:
        "results/subret_recordings.csv"
    log:
        "logs/add_data.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/add_types_to_table.py"


rule convert_to_nwb:
    input:
        "results/subret_recordings.csv",
        "workflow/sheets/openfield_cells.csv",
        "workflow/sheets/muscimol_cells.csv"
    output:
        "results/openfield_nwb.csv",
        "results/openfield_cells_nwb.csv",
        "results/muscimol_cells_nwb.csv"
    log:
        "logs/convert_openfield_to_nwb.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/convert_to_nwb.py"


rule convert_tmaze:
    input:
        "results/subret_recordings.csv",
        "workflow/sheets/tmaze_times.csv"
    output:
        "results/tmaze_times_nwb.csv"
    log:
        "logs/convert_tmaze.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/convert_tmaze.py"


rule convert_all_data:
    input:
        "results/subret_recordings.csv",
        "results/processed_nwbfiles.csv",
    output:
        "results/other_converted.csv",
        "results/other_process.csv",
        "results/index_temp.csv"
    log:
        "logs/convert_all_data.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/convert_remaining_data.py"

rule remove_temp:
    input:
        "results/index_temp.csv"
    output:
        "results/index.csv"
    log:
        "logs/remove_temp.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/remove_temp.py"