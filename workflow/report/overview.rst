========
Overview
========

This document serves as an overview of the ATN SUB snakemake workflow

Preprocessing
=============

- Firstly, all the axona files on disk are processed into a CSV describing the files and their location on the disk. For instance, the type of trial is described.
- Then, using a set of parameter files in this repository, some extra information is described about the files, and a double check is performed to ensure the correct trial type was inferred for the data that will be used.
- For all trials that had a type set in the previous step (marking that they are intended to be used for analysis), these files are converted into the NWB 2.0 format.

Analysis
========

- The NWB files from the previous step are processed to include:
    - Normalised LFP that is filtered between 1 and 100Hz. Normalising is performed by subtracting the average value of each signal from the signal, and the dividing the signal by the standard deviation of the signal. If the standard deviation is 0, only the average is subtracted.
    - An average signal per brain region recorded from. This average signal can be calculated as the average over all signals from that brain region, or a select subset based on a configuration. In this study, the average signal is selected to be from a set of bipolar wires as they are thicker and record LFP signal well.
    - Power spectral analysis (Welch) for each of the normalised LFP signals (including the synthetic LFP signal). Power spectra are calculated in decibels, relative to the maximum PSD value. The number of segments are 250 and averaging is performed with the mean.
    - Finally, signals are marked as "clean" or "outliers". This marking is somewhat experimental, but is based on z-scores of the signals. The overall average signal and standard deviation of the signals at each time point are calculated. The z-score of each signal is computed by subtracting the average signal and dividing the standard deviation of the signals at each time point. Signals which have a median z-score above a certain threshold are marked as outliers (1.2 by default).


Plotting
========

- The data is then plotted; for openfield data the following is produced:
    - A comparison for each session between calculating PSDS using; A - the average of each individual PSD from that brain region (clean vs outlier). B - The PSD of the average signal. C - The confidence interval of PSDs over only clean signals.
    - Two summary plots for each brain region and PSD calculation type (signal average, then PSD vs PSD calculation then average PSDs). This involves a per-rat CI plot, and an overall control vs Lesion group comparison.

Observations
============

- There is little difference between using the average of the two bipolar wires versus all the signals on an overall level. However, on a per session basis there can be some differences, though they are usually small and involve magnitude differences as opposed to shape.