# The Chemical Checker Protocols Repository

The **Chemical Checker (CC)** is a resource of small molecule signatures. In the CC, compounds are described from multiple viewpoints, spanning every aspect of the drug discovery pipeline, from chemical properties to clinical outcomes. Bioactivity signatures dynamically evolve with new data and processing strategies. This repository presents a Python package to modify and/or generate novel bioactivity spaces and signatures, describing the main steps needed to leverage diverse bioactivity data with the current knowledge, as catalogued in the Chemical Checker, using the predefined data curation pipeline.

## Table of Contents
1. [The Chemical Checker](#the-chemical-checker)
2. [The Signaturizers](#the-signaturizers)
3. [CC Protocols Publication](#cc-protocols-publication)
4. [Repository Structure](#repository-structure)

## The Chemical Checker
For a quick exploration of what the CC enables, please visit the [CC web app](http://chemicalchecker.org).

To explore the CC main repository, please visit its [Gitlab repository](https://gitlabsbnb.irbbarcelona.org/packages/chemical_checker).

For full documentation of the Python package, please see the [Documentation](http://packages.sbnb-pages.irbbarcelona.org/chemical_checker).

Concepts and methods are best described in the original CC publication, [Duran-Frigola et al. 2019](https://biorxiv.org/content/10.1101/745703v1).

## The Signaturizers
To explore the Signaturizers repository (i.e., to generate CC signatures for any chemical compound of interest), please visit the [original version](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer) or the [latest and stereochemically-aware models](https://gitlabsbnb.irbbarcelona.org/packages/signaturizer3d).

## CC Protocols Publication
Detailed explanations of the CC Protocols are best described in the corresponding publication, [*Comajuncosa-Creus et al. 2024*](https://google.com).

## Repository Structure
In the **Chemical Checker Protocols Repository**, we illustrate the functioning of the protocol through four specific examples, including:
- The incorporation of new compounds into an already existing bioactivity space (B1.002).
- A change in the data pre-processing without altering the underlying experimental data (D1.002).
- The creation of two novel bioactivity spaces from scratch (D6.001 and M1.001).

### Folders and Files
- `notebooks`: iPython notebooks (4) for the integration of new bioactivity data using the defined data curation pipeline.
- `data`: Input bioactivity data.

The local directories of the CC are divided in full and reference sets of compounds. The full directory contains the computed signatures (from 0 to III) of the complete sets of small molecules for each CC space. The reference set includes a non-redundant subset of the data, computed using the distance matrix among all compounds. 

