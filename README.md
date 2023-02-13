# Snakemake workflow: ATNx

[![Snakemake](https://img.shields.io/badge/snakemake-≥6.3.0-brightgreen.svg)](https://snakemake.github.io)

A Snakemake workflow for an ATN lesion experiment.

## Installation

See `envs/nwb_simuran.yml` for a conda environment, or install pip requirements as `pip install -r requirements.txt`.

## Usage

This project can be used to transform raw Axona data into processed NWB files, as well as work with those processed NWB files to perform analysis related to the ATN lesion experiment and LFP signals.

The usage of this workflow is described in the [Snakemake Workflow Catalog](https://snakemake.github.io/snakemake-workflow-catalog/?usage=seankmartin/atn-sub-lfp-workflow).
See the [README in config](config/README.md) for more information on possible parameters.

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
