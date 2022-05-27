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