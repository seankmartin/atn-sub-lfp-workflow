# Configuration

## The following config files are available, and likely need to be modified

### snakemake_config.yml

This file contains the following variables, in particular, 1 and 2 liekly need to be modified:

1. data_directory: The directory where the SUB data is stored.
2. ca1_directory: The directory where the CA1 data is stored.
3. simuran_config: The path to the simuran config file (simuran_params.yml).
4. openfield_filter: The filter to use for openfield recordings (openfield_recordings.yml).
5. tmaze_filter: The filter to use for tmaze recordings (tmaze_recordings.yml).
6. overwrite_nwb: Whether to overwrite the NWB files if they already exist (False).
7. sleep_only: Whether to only process sleep recordings (False).
8. overwrite_sleep: Whether to overwrite the sleep analysis files if they already exist (False).
9. except_nwb_errors: Whether to ignore NWB errors (True).

### simuran_params.yml

This file contains individual parameters for each analysis, such as the band to use to consider theta to be in (e.g. 6-12 Hz):

1. cfg_base_dir: The base directory to use for data referred to by relative paths in the config files.
2. do_spectrogram_plot: Whether to plot the spectrogram (False).
3. plot_psd: Whether to plot the power spectrum (True).
4. image_format: The format to use for images (png).
5. loader: The name of the loader to use (neurochat).
6. loader_kwargs: The keyword arguments to pass to the loader.
7. clean_method: The method to use to clean the LFP signals, by default it zscore normalises the signals and the picks the bipolar electrode signals from these if they don't exceed a standard deviation from the average for non-canulated rats. For canulated rats, it proceeds similarly but uses all clean signals, not just those on the bipolar electrodes.
8. clean_kwargs: The keyword arguments to pass to the clean method for non-canulated rats.
9. can_clean_kwargs: The keyword arguments to pass to the clean method for canulated rats.
10. z_score_threshold: The z-score threshold to use for the LFP cleaning.
11. fmin: The minimum frequency to consider for filtering.
12. fmax: The maximum frequency to consider for filtering.
13. filter_kwargs: The keyword arguments to pass to the filter method.
14. theta_min, theta_max: The minimum and maximum frequencies to consider for theta.
15. delta_min, delta_max: The minimum and maximum frequencies to consider for delta.
16. low_gamma_min, low_gamma_max: The minimum and maximum frequencies to consider for low_gamma.
17. high_gamma_min, high_gamma_max: The minimum and maximum frequencies to consider for high_gamma.
18. beta_min, beta_max: The minimum and maximum frequencies to consider for beta.
19. psd_scale: The scale to use for the power spectrum (decibels or volts).
20. number_of_shuffles_sta: The number of shuffles of time to use for the STA analysis.
21. num_spike_shuffles: The number of shuffles of spikes to use for the STA analysis.
22. max_psd_freq, max_fooof_freq: The maximum frequency to consider for the power spectrum and fooof analysis.
23. speed_theta_samples_per_second: How many speed theta samples per second to use, as this data is binned.
24. max_speed: The maximum speed to consider for the speed theta analysis, in cm/s.
25. tmaze_minf, tmaze_maxf: The minimum and maximum frequencies to consider for the tmaze analysis.
26. tmaze_winsec: The window size to use for the tmaze analysis, in seconds for LFP analysis.
27. max_lfp_lengths: How to split up the LFP signal during tmaze analysis. Defaults give 1 second windows.
28. tmaze_egf: Whether to use eeg or egf (higher rate signal) in tmaze anlaysis.
29. spindles_use_avg: Whether to run spindle analysis on the average signal or on all signals.
30. use_first_two_for_ripples: Whether to use the first two signals for ripple analysis or all signals.
31. lfp_ripple_rate: the rate of the high frequency LFP signal to use for ripple analysis, can be a downsample of the full egf rate.
32. min_sleep_length: The minimum length of sleep to consider for sleep analysis, in seconds.
33. only_kay_detect: Whether to only use Kay's algorithm for sleep detection.
34. except_nwb_errors: Whether to ignore NWB errors (True).
35. sleep_join_tol: The allowed time of movement between sleep epochs to join them, in seconds (0.0).
36. sleep_max_interval_size: The maximum allowed time for a sleep epoch to be, before splitting for efficiency, in seconds (300).

## The additional config files are (unlikely to require modification)

### tmaze_recordings.yml

This lists how to obtain the tmaze recordings, with 8 control rats and 6 lesion rats considered.

### openfield_recordings.yml

This lists the name of the rats to use openfield recordings for, with 6 control rats and 5 lesion rats considered.
