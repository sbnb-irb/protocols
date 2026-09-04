# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Imports
import os
import sys
import numpy as np

# Make sure cc_pipeline.py / utils.py (placed alongside this notebook) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath('.')))

from cc_pipeline import (
    default_dataset_specs,
    load_dataset_dataframes,
    get_cc_universe,
    report_universe_overlap,
    diagnose_and_plot,
    report_minmax,
    fit_sign0,
    fit_sign1,
    fit_sign2,
    fit_sign3,
)
from utils import generate_log_filename, get_basename, setup_logging

# Log to both a timestamped file (under ./logs) and this notebook's output,
# using the same conventions as the HPC script and the rest of the project.
setup_logging(generate_log_filename("logs", suffix=get_basename("pertprot_interactive")))

# Define variables and paths
os.environ['CC_CONFIG'] = '/scratch/sbnb/sayala/chemical_checker/setup/cc_config.json'
local_cc_dir = '/scratch/sbnb/sayala/protocols/local_CC_D6'
PATH_TO_DATA = "/scratch/sbnb/sayala/cc_data/data"  # See Download_Data.ipynb // Procedure step 3
PERTURBPROT_DATA_DIR = "/scratch/sbnb/sayala/protocols/data/PerturbProt"

# CC import
from chemicalchecker import ChemicalChecker

# Apply general settings
# %matplotlib inline
os.makedirs(PATH_TO_DATA, exist_ok=True)
ChemicalChecker.set_verbosity('DEBUG')  # CRITICAL, ERROR, WARN, INFO or DEBUG
cc_local = ChemicalChecker(local_cc_dir, dbconnect=False, custom_data_path=PATH_TO_DATA)

# Set sanitizer parameters to avoid expensive filtering that might cause issues
sanitizer_kwargs = {
    'chunk_size': 500000,   # adjust chunk size for memory
}

# Optional InChIKey -> InChI mapping dict for sign3.
mapp = None
"""
# Alternatively, you can provide your in-house python dictionary to map InchiKey's to InChI's (see example below)
mapp = {
'LPXQRXLUHJKZIE-UHFFFAOYSA-N': 'InChI=1S/C4H4N6O/c5-4-6-2-1(3(11)7-4)8-10-9-2/h(H4,5,6,7,8,9,10,11)',
'BZKPWHYZMXOIDC-UHFFFAOYSA-N': 'InChI=1S/C4H6N4O3S2/c1-2(9)6-3-7-8-4(12-3)13(5,10)11/h1H3,(H2,5,10,11)(H,6,7,9)',
'XZWYZXLIPXDOLR-UHFFFAOYSA-N': 'InChI=1S/C4H11N5/c1-9(2)4(7)8-3(5)6/h1-2H3,(H5,5,6,7,8)'
}
"""

# %% [markdown]
# ## LOAD INPUT DATA -- Perturbation proteomics datasets

# %%
# Build the dataset registry (name, D6 code, csv path) and load all dataframes.
# specs is a dict keyed by 'dcmoa', 'dcmoa_deps', 'deps', 'shared_deps', 'allprot', 'shared_allprot'
specs = {s.key: s for s in default_dataset_specs(PERTURBPROT_DATA_DIR)}
load_dataset_dataframes(specs.values())

# %% [markdown]
# ### All Proteins

# %%
print(specs['allprot'].df.shape)
specs['allprot'].df.head()

# %% [markdown]
# ### All Shared Proteins

# %%
print(specs['shared_allprot'].df.shape)
specs['shared_allprot'].df.head()

# %% [markdown]
# ### DEPs

# %%
print(specs['deps'].df.shape)
specs['deps'].df.head()

# %% [markdown]
# ### Shared DEPs

# %%
print(specs['shared_deps'].df.shape)
specs['shared_deps'].df.head()

# %% [markdown]
# ### DeepCoverMoa

# %%
print(specs['dcmoa'].df.shape)
specs['dcmoa'].df.head()

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
print(specs['dcmoa_deps'].df.shape)
specs['dcmoa_deps'].df.head()

# %% [markdown]
# ## sign0

# %% [markdown]
# ### DeepCoverMoa

# %%
sign0_dcmoa = fit_sign0(cc_local, specs['dcmoa'].code, specs['dcmoa'].df, sanitizer_kwargs)
sign0_dcmoa.shape

# %%
diagnose_and_plot(sign0_dcmoa)
report_minmax(sign0_dcmoa)

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
sign0_dcmoa_deps = fit_sign0(cc_local, specs['dcmoa_deps'].code, specs['dcmoa_deps'].df, sanitizer_kwargs)
sign0_dcmoa_deps.shape

# %%
diagnose_and_plot(sign0_dcmoa_deps)
report_minmax(sign0_dcmoa_deps)

# %% [markdown]
# ### DEPs

# %%
sign0_deps = fit_sign0(cc_local, specs['deps'].code, specs['deps'].df, sanitizer_kwargs)
sign0_deps.shape

# %%
diagnose_and_plot(sign0_deps)
report_minmax(sign0_deps)

# %% [markdown]
# ### Shared DEPs

# %%
sign0_shared_deps = fit_sign0(cc_local, specs['shared_deps'].code, specs['shared_deps'].df, sanitizer_kwargs)
sign0_shared_deps.shape

# %%
diagnose_and_plot(sign0_shared_deps)
report_minmax(sign0_shared_deps)

# %% [markdown]
# ### AllProt

# %%
sign0_allprot = fit_sign0(cc_local, specs['allprot'].code, specs['allprot'].df, sanitizer_kwargs)
sign0_allprot.shape

# %%
diagnose_and_plot(sign0_allprot)
report_minmax(sign0_allprot)

# %% [markdown]
# ### All Shared Proteins

# %%
sign0_shared_allprot = fit_sign0(cc_local, specs['shared_allprot'].code, specs['shared_allprot'].df, sanitizer_kwargs)
sign0_shared_allprot.shape

# %%
diagnose_and_plot(sign0_shared_allprot)
report_minmax(sign0_shared_allprot)

# %% [markdown]
# ## sign1

# %% [markdown]
# ### DeepCoverMoa

# %%
sign1_dcmoa, neig1_dcmoa = fit_sign1(cc_local, specs['dcmoa'].code, sign0_dcmoa)
sign1_dcmoa.shape, neig1_dcmoa.shape

# %%
diagnose_and_plot(sign1_dcmoa)
report_minmax(sign1_dcmoa)

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
sign1_dcmoa_deps, neig1_dcmoa_deps = fit_sign1(cc_local, specs['dcmoa_deps'].code, sign0_dcmoa_deps)
sign1_dcmoa_deps.shape, neig1_dcmoa_deps.shape

# %%
diagnose_and_plot(sign1_dcmoa_deps)
report_minmax(sign1_dcmoa_deps)

# %% [markdown]
# ### DEPs

# %%
sign1_deps, neig1_deps = fit_sign1(cc_local, specs['deps'].code, sign0_deps)
sign1_deps.shape, neig1_deps.shape

# %%
diagnose_and_plot(sign1_deps)
report_minmax(sign1_deps)

# %% [markdown]
# ### Shared DEPs

# %%
sign1_shared_deps, neig1_shared_deps = fit_sign1(cc_local, specs['shared_deps'].code, sign0_shared_deps)
sign1_shared_deps.shape, neig1_shared_deps.shape

# %%
diagnose_and_plot(sign1_shared_deps)
report_minmax(sign1_shared_deps)

# %% [markdown]
# ### All Proteins

# %%
sign1_allprot, neig1_allprot = fit_sign1(cc_local, specs['allprot'].code, sign0_allprot)
sign1_allprot.shape, neig1_allprot.shape

# %%
diagnose_and_plot(sign1_allprot)
report_minmax(sign1_allprot)

# %% [markdown]
# ### All Shared Proteins

# %%
sign1_shared_allprot, neig1_shared_allprot = fit_sign1(cc_local, specs['shared_allprot'].code, sign0_shared_allprot)
sign1_shared_allprot.shape, neig1_shared_allprot.shape

# %%
diagnose_and_plot(sign1_shared_allprot)
report_minmax(sign1_shared_allprot)

# %% [markdown]
# ## sign2

# %% [markdown]
# ### DeepCoverMoa

# %%
sign2_dcmoa = fit_sign2(cc_local, specs['dcmoa'].code, sign1_dcmoa, neig1_dcmoa)
sign2_dcmoa.shape

# %%
diagnose_and_plot(sign2_dcmoa)
report_minmax(sign2_dcmoa)

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
sign2_dcmoa_deps = fit_sign2(cc_local, specs['dcmoa_deps'].code, sign1_dcmoa_deps, neig1_dcmoa_deps)
sign2_dcmoa_deps.shape

# %%
diagnose_and_plot(sign2_dcmoa_deps)
report_minmax(sign2_dcmoa_deps)

# %% [markdown]
# ### DEPs

# %%
sign2_deps = fit_sign2(cc_local, specs['deps'].code, sign1_deps, neig1_deps)
sign2_deps.shape

# %%
diagnose_and_plot(sign2_deps)
report_minmax(sign2_deps)

# %% [markdown]
# ### Shared DEPs

# %%
sign2_shared_deps = fit_sign2(cc_local, specs['shared_deps'].code, sign1_shared_deps, neig1_shared_deps)
sign2_shared_deps.shape

# %%
diagnose_and_plot(sign2_shared_deps)
report_minmax(sign2_shared_deps)

# %% [markdown]
# ### All Proteins

# %%
sign2_allprot = fit_sign2(cc_local, specs['allprot'].code, sign1_allprot, neig1_allprot)
sign2_allprot.shape

# %%
diagnose_and_plot(sign2_allprot)
report_minmax(sign2_allprot)

# %% [markdown]
# ### All Shared Proteins

# %%
sign2_shared_allprot = fit_sign2(cc_local, specs['shared_allprot'].code, sign1_shared_allprot, neig1_shared_allprot)
sign2_shared_allprot.shape

# %%
diagnose_and_plot(sign2_shared_allprot)
report_minmax(sign2_shared_allprot)

# %% [markdown]
# ## sign3

# %%
# CC universe (the ~25 canonical CC spaces) -- computed once, reused for every dataset below
cc_universe = get_cc_universe(cc_local)

# %% [markdown]
# ### DeepCoverMoa

# %%
report_universe_overlap('DeepCoverMoa', sign2_dcmoa, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_dcmoa = fit_sign3(cc_local, specs['dcmoa'].code, sign2_dcmoa, sign1_dcmoa, mapping_dict=mapp)
np.array(sign3_dcmoa).shape

# %%
diagnose_and_plot(sign3_dcmoa, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_dcmoa)

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
report_universe_overlap('DeepCoverMoa DEPs', sign2_dcmoa_deps, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_dcmoa_deps = fit_sign3(cc_local, specs['dcmoa_deps'].code, sign2_dcmoa_deps, sign1_dcmoa_deps, mapping_dict=mapp)
np.array(sign3_dcmoa_deps).shape

# %%
diagnose_and_plot(sign3_dcmoa_deps, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_dcmoa_deps)

# %% [markdown]
# ### DEPs

# %%
report_universe_overlap('DEPs', sign2_deps, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_deps = fit_sign3(cc_local, specs['deps'].code, sign2_deps, sign1_deps, mapping_dict=mapp)
np.array(sign3_deps).shape

# %%
diagnose_and_plot(sign3_deps, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_deps)

# %% [markdown]
# ### Shared DEPs

# %%
report_universe_overlap('Shared DEPs', sign2_shared_deps, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_shared_deps = fit_sign3(cc_local, specs['shared_deps'].code, sign2_shared_deps, sign1_shared_deps, mapping_dict=mapp)
np.array(sign3_shared_deps).shape

# %%
diagnose_and_plot(sign3_shared_deps, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_shared_deps)

# %% [markdown]
# ### AllProt

# %%
report_universe_overlap('AllProt', sign2_allprot, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_allprot = fit_sign3(cc_local, specs['allprot'].code, sign2_allprot, sign1_allprot, mapping_dict=mapp)
np.array(sign3_allprot).shape

# %%
diagnose_and_plot(sign3_allprot, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_allprot)

# %% [markdown]
# ### All Shared Proteins

# %%
report_universe_overlap('All Shared Proteins', sign2_shared_allprot, cc_universe)

# %%
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_shared_allprot = fit_sign3(cc_local, specs['shared_allprot'].code, sign2_shared_allprot, sign1_shared_allprot, mapping_dict=mapp)
np.array(sign3_shared_allprot).shape

# %%
diagnose_and_plot(sign3_shared_allprot, sizes=('medium', 'small'), ref_cctype='sign3')
report_minmax(sign3_shared_allprot)
