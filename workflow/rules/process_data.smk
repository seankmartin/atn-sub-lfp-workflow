rule preprocess_data:
    output:
        "results/axona_file_index.csv"
    script:
        "scripts/index_axona_files.py"

rule add_data_types:
    input:
        "results/axona_file_index.csv"
    output:
        "results/subret_recordings.csv"
    script:
        "python"