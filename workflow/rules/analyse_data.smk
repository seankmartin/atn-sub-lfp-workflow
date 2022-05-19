rule process_openfield_lfp:
    input:
        "results/subret_recordings.csv"
    output:
        "results/sim_results--subret_recordings.csv"
    params:
        function_path="scripts/functions/fn_spectra.py"
    script:
        "../scripts/process_openfield_lfp.py"