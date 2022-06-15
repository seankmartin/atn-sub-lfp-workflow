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
    output:
        "results/openfield_nwb.csv",
        "results/openfield_cells_nwb.csv",
    log:
        "logs/convert_openfield_to_nwb.log"
    conda:
        "../../envs/nwb_simuran.yml"
    script:
        "../scripts/convert_to_nwb.py"