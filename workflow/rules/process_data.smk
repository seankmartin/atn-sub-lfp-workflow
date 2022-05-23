rule preprocess_data:
    output:
        "results/axona_file_index.csv"
    script:
        "../scripts/index_axona_files.py"

rule add_data_types:
    input:
        "results/axona_file_index.csv"
    output:
        "results/subret_recordings.csv"
    script:
        "../scripts/add_types_to_table.py"

rule openfield_to_nwb:
    input:
        "results/subret_recordings.csv"
    output:
        "results/openfield_nwb.csv"
    script:
        "../scripts/convert_to_nwb.py"