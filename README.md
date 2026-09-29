# The Chemical Checker Protocols Repository

The **Chemical Checker (CC)** is a resource of small molecule signatures. In the CC, compounds are described from multiple viewpoints, spanning every aspect of the drug discovery pipeline, from chemical properties to clinical outcomes. Bioactivity signatures dynamically evolve with new data and processing strategies. This repository presents a Python package to modify and/or generate novel bioactivity spaces and signatures, describing the main steps needed to leverage diverse bioactivity data with the current knowledge, as catalogued in the Chemical Checker, using the predefined data curation pipeline.

## Table of Contents
1. [The Chemical Checker](#the-chemical-checker)
2. [The Signaturizers](#the-signaturizers)
3. [CC Protocols Publication](#cc-protocols-publication)
4. [Repository Structure](#repository-structure)
5. [Command-line Toolkit](#command-line-toolkit)

## The Chemical Checker
For a quick exploration of what the CC enables, please visit the [CC web app](http://chemicalchecker.org).

To explore the CC main repository, please visit its [Gitlab repository](https://gitlabsbnb.irbbarcelona.org/packages/chemical_checker).

For full documentation of the Python package, please see the [Documentation](http://packages.sbnb-pages.irbbarcelona.org/chemical_checker).

Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://biorxiv.org/content/10.1101/745703v1).

## The Signaturizers
To explore the Signaturizers repository (i.e., to generate CC signatures for any chemical compound of interest), please visit the [original version](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer) or the [latest and stereochemically-aware models](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer3d).

## CC Protocols Publication
Detailed explanations of the CC Protocols are best described in the corresponding publication, [*Comajuncosa-Creus et al. 2024*](https://www.biorxiv.org/content/10.1101/2024.12.04.626832v1).

## Repository Structure
In the **Chemical Checker Protocols Repository**, we illustrate the functioning of the protocol through four specific examples, including:
- The incorporation of new compounds into an already existing bioactivity space (B1.002).
- A change in the data pre-processing without altering the underlying experimental data (D1.002).
- The creation of two novel bioactivity spaces from scratch (D6.001 and M1.001).

### Folders and Files
- `notebooks`: iPython notebooks (4) for the integration of new bioactivity data using the defined data curation pipeline.
- `data`: links to download the preprocessed bioactivity data to reproduce the results in the manuscript.
- `src/chemcheck_protocols`: the same pipeline as a Python package and command-line tool (see below).
- `configs`: run configurations for the toolkit; `configs/paper_tasks` reproduces the manuscript examples.
- `scripts/chemcheck_exec.sh`: runs any command inside the CC Singularity container, directly or as a SLURM job.
- `tests`: tests for the toolkit (`python -m pytest`).

The generated local directories of the CC are divided in full and reference sets of compounds. The full directory contains the computed signatures (from 0 to III) of the complete sets of small molecules for each CC space. The reference set includes a non-redundant subset of the data, computed using the distance matrix among all compounds. 

## Command-line Toolkit
`chemcheck_protocols` runs the notebooks' pipeline (sign0 -> sign1 -> sign2 -> sign3) from a YAML
run configuration, with validation, logging and the option to run each stage separately. It needs
the Python environment of the CC Singularity image; `scripts/chemcheck_exec.sh` runs commands in it.

**1. Configure the container** (once per shell; see the header of `scripts/chemcheck_exec.sh`):
```bash
export CC_IMAGE=/path/to/cc.simg                  # Singularity image with chemicalchecker
export CC_CONFIG=/path/to/cc_config.json          # or cc_config: in the run configuration
export CC_BIND=/path/to/data,/path/to/local_CC    # folders the container must see
```

**2. Write a run configuration**, e.g. `configs/paper_tasks/m1_001.yaml` (after downloading the data
listed in `data/DATA.README`). Relative paths are resolved against the configuration file:
```yaml
cc_root: ../../local_CC_M1              # local CC instance, with the reference CC signatures
datasets:
  - key: m1
    name: Drug-microbiome
    dataset_code: M1.001
    source: {format: wide_matrix, path: ../../data/M1/microbiota_raw.csv}
    reference_spaces: new_space         # or {extends: B1.001} for a new version of a CC space
```

**3. Fit the signatures.** sign0-sign2 take minutes and can run interactively; sign3 takes hours,
so on a cluster submit it as a job. `--start-stage sign3` reuses the sign1/sign2 already fitted:
```bash
bash   scripts/chemcheck_exec.sh python -m chemcheck_protocols fit-signatures \
       --config configs/paper_tasks/m1_001.yaml --max-stage sign2
sbatch scripts/chemcheck_exec.sh python -m chemcheck_protocols fit-signatures \
       --config configs/paper_tasks/m1_001.yaml --start-stage sign3
```
SLURM resources default to the `#SBATCH` lines of the script; override them per job with sbatch
flags (e.g. `sbatch --partition=gpu --gres=gpu:1 --time=3-00:00:00 scripts/chemcheck_exec.sh ...`).
Run `python -m chemcheck_protocols fit-signatures --help` for all options.
