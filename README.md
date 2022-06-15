# Snakemake workflow: ATNx

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)
[![GitHub actions status](https://github.com/seankmartin/atn-sub-lfp-workflow/workflows/Tests/badge.svg?branch=main)](https://github.com/seankmartin/atn-sub-lfp-workflow/actions?query=branch%3Amain+workflow%3ATests)

A Snakemake workflow for an ATN lesion experiment.

## Requirements

See `envs/condaenv.yml`. Requires pynwb > 2 and simuran. TODO add fixed simuran to requirements.

## Usage

The usage of this workflow is described in the [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/?usage=<owner>%2F<repo>).

If you use this workflow in a paper, don't forget to give credits to the authors by citing the URL of this (original) repository and its DOI (see above).

## NWB file layout

General metadata is available about acquisition information and experimenter information.

### Top level tables

- electrodes
- units

### Processing modules

#### behaviour

- running_speed: time series
- CompassDirection : heading direction information
- Position : camera tracking information

#### average_lfp

- A single LFP signal per brain region

#### ecehpys and normalised_lfp

- LFP : LFP data

#### tables

- lfp_coherence, coherence_table
- lfp_power, power_spectra
- speed_theta, speed_lfp_table
