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
import numpy as np
import pandas as pd

# Define variables and paths
os.environ['CC_CONFIG'] = '/scratch/sbnb/sayala/chemical_checker/setup/cc_config.json'
local_cc_dir = '/scratch/sbnb/sayala/protocols/local_CC_D6'
PATH_TO_DATA = "/scratch/sbnb/sayala/cc_data/data"  # See Download_Data.ipynb // Procedure step 3

# CC import
from chemicalchecker import ChemicalChecker

# Apply general settings
# %matplotlib inline
os.makedirs(PATH_TO_DATA, exist_ok=True)
ChemicalChecker.set_verbosity('DEBUG') # CRITICAL, ERROR, WARN, INFO or DEBUG
cc_local = ChemicalChecker(local_cc_dir, dbconnect=False, custom_data_path=PATH_TO_DATA)

# %% [markdown]
# ## LOAD INPUT DATA -- Perturbation proteomics datasets

# %%
# Load the raw binary data
# Rows: compounds
# Columns: Uniprot IDs
all_proteins_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/all_studies_wide.csv"
all_proteins_shared_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/all_studies_wide_shared.csv"
deps_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/all_studies_deps_wide.csv"
deps_shared_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/all_studies_deps_wide_shared.csv"
deepcovermoa_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/deepcovermoa_wide.csv"
deepcovermoa_deps_f = "/scratch/sbnb/sayala/protocols/data/PerturbProt/deepcovermoa_deps_wide.csv"

df_allprot = pd.read_csv(all_proteins_f, sep=',', index_col=0)
df_allprot_shared = pd.read_csv(all_proteins_shared_f, sep=',', index_col=0)
df_deps = pd.read_csv(deps_f, sep=',', index_col=0)
df_shared_deps = pd.read_csv(deps_shared_f, sep=',', index_col=0)
df_deepcovermoa = pd.read_csv(deepcovermoa_f, sep=',', index_col=0)
df_deepcovermoa_deps = pd.read_csv(deepcovermoa_deps_f, sep=',', index_col=0)
# %% [markdown]
# ### All Proteins

# %%
print(df_allprot.shape)
df_allprot.head()

# %% [markdown]
# ### All Shared Proteins

# %%
print(df_allprot_shared.shape)
df_allprot_shared.head()

# %% [markdown]
# ### DEPs

# %%
print(df_deps.shape)
df_deps.head()

# %% [markdown]
# ### Shared DEPs

# %%
print(df_shared_deps.shape)
df_shared_deps.head()

# %% [markdown]
# ### DeepCoverMoa 

# %%
print(df_deepcovermoa.shape)
df_deepcovermoa.head()

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
print(df_deepcovermoa_deps.shape)
df_deepcovermoa_deps.head()

# %% [markdown]
# ## sign0

# %%
# Dataset Names
dataset_dcmoa = 'D6.002'
dataset_dcmoa_deps = 'D6.003' 
dataset_deps = 'D6.004'
dataset_shared_deps = 'D6.005'
dataset_allprot = 'D6.006'
dataset_shared_allprot = 'D6.007'

# Set sanitizer parameters to avoid expensive filtering that might cause issues
sanitizer_kwargs = {
    'chunk_size': 500000         # adjust chunk size for memory
}

# Instantiation of sign0 data structures for the new spaces: full and references
sign0_dcmoa = cc_local.signature(dataset_dcmoa, 'sign0')
sign0_dcmoa_deps = cc_local.signature(dataset_dcmoa_deps, 'sign0')
sign0_deps = cc_local.signature(dataset_deps, 'sign0')
sign0_shared_deps = cc_local.signature(dataset_shared_deps, 'sign0')
sign0_allprot = cc_local.signature(dataset_allprot, 'sign0')
sign0_shared_allprot = cc_local.signature(dataset_shared_allprot, 'sign0')

# %% [markdown]
# ### DeepCoverMoa

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_dcmoa.clear_all()

# Fit sign0
sign0_dcmoa.fit(X=df_deepcovermoa.values, keys=list(df_deepcovermoa.index), features=list(df_deepcovermoa.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_dcmoa.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_dcmoa = sign0_dcmoa.diagnosis()

# Plot small diagnosis plots
diag0_dcmoa.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_dcmoa).flatten()), np.max(np.array(sign0_dcmoa).flatten())

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_dcmoa_deps.clear_all()

# Fit sign0
sign0_dcmoa_deps.fit(X=df_deepcovermoa_deps.values, keys=list(df_deepcovermoa_deps.index), features=list(df_deepcovermoa_deps.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_dcmoa_deps.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_dcmoa_deps = sign0_dcmoa_deps.diagnosis()

# Plot small diagnosis plots
diag0_dcmoa_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_dcmoa_deps).flatten()), np.max(np.array(sign0_dcmoa_deps).flatten())

# %% [markdown]
# ### DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_deps.clear_all()

# Fit sign0
sign0_deps.fit(X=df_deps.values, keys=list(df_deps.index), features=list(df_deps.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_deps.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_deps = sign0_deps.diagnosis()

# Plot small diagnosis plots
diag0_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_deps).flatten()), np.max(np.array(sign0_deps).flatten())

# %% [markdown]
# ### Shared DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_shared_deps.clear_all()

# Fit sign0
sign0_shared_deps.fit(X=df_shared_deps.values, keys=list(df_shared_deps.index), features=list(df_shared_deps.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_shared_deps.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_shared_deps = sign0_shared_deps.diagnosis()

# Plot small diagnosis plots
diag0_shared_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_shared_deps).flatten()), np.max(np.array(sign0_shared_deps).flatten())

# %% [markdown]
# ### AllProt

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_allprot.clear_all()

# Fit sign0
sign0_allprot.fit(X=df_allprot.values, keys=list(df_allprot.index), features=list(df_allprot.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_allprot.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_allprot = sign0_allprot.diagnosis()

# Plot small diagnosis plots
diag0_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_allprot).flatten()), np.max(np.array(sign0_allprot).flatten())

# %% [markdown]
# ### All Shared Proteins

# %%
# Cleaning both full and reference datasets. This is crucial!
sign0_shared_allprot.clear_all()

# Fit sign0
sign0_shared_allprot.fit(X=df_allprot_shared.values, keys=list(df_allprot_shared.index), features=list(df_allprot_shared.columns), sanitizer_kwargs=sanitizer_kwargs)

# %%
sign0_shared_allprot.shape

# %%
# Instantiation of diag0 (diagnosis plots)
diag0_shared_allprot = sign0_shared_allprot.diagnosis()

# Plot small diagnosis plots
diag0_shared_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0_shared_allprot).flatten()), np.max(np.array(sign0_shared_allprot).flatten())

# %% [markdown]
# ## sign1

# %%
# Instantiation of sign1 data structures for the new spaces: full and references
sign1_dcmoa = cc_local.signature(dataset_dcmoa, 'sign1')
sign1_dcmoa_deps = cc_local.signature(dataset_dcmoa_deps, 'sign1')
sign1_deps = cc_local.signature(dataset_deps, 'sign1')
sign1_shared_deps = cc_local.signature(dataset_shared_deps, 'sign1')
sign1_allprot = cc_local.signature(dataset_allprot, 'sign1')
sign1_shared_allprot = cc_local.signature(dataset_shared_allprot, 'sign1')

# %% [markdown]
# ### DeepCoverMoa

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_dcmoa.clear_all()

# Fitting sign1
sign1_dcmoa.fit(sign0_dcmoa)

# %%
sign1_dcmoa.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_dcmoa = sign1_dcmoa.diagnosis()

# Plot medium & small diagnosis plots
diag1_dcmoa.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_dcmoa = cc_local.get_signature("neig1", "full", dataset_dcmoa)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_dcmoa.clear_all()

# Fitting neig1
neig1_dcmoa.fit(sign1_dcmoa)

# %%
neig1_dcmoa.shape

# %%
np.min(np.array(sign1_dcmoa).flatten()), np.max(np.array(sign1_dcmoa).flatten())

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_dcmoa_deps.clear_all()

# Fitting sign1
sign1_dcmoa_deps.fit(sign0_dcmoa_deps)

# %%
sign1_dcmoa_deps.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_dcmoa_deps = sign1_dcmoa_deps.diagnosis()

# Plot medium & small diagnosis plots
diag1_dcmoa_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_dcmoa_deps = cc_local.get_signature("neig1", "full", dataset_dcmoa_deps)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_dcmoa_deps.clear_all()

# Fitting neig1
neig1_dcmoa_deps.fit(sign1_dcmoa_deps)

# %%
neig1_dcmoa_deps.shape

# %%
np.min(np.array(sign1_dcmoa_deps).flatten()), np.max(np.array(sign1_dcmoa_deps).flatten())

# %% [markdown]
# ### DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_deps.clear_all()

# Fitting sign1
sign1_deps.fit(sign0_deps)

# %%
sign1_deps.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_deps = sign1_deps.diagnosis()

# Plot medium & small diagnosis plots
diag1_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_deps = cc_local.get_signature("neig1", "full", dataset_deps)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_deps.clear_all()

# Fitting neig1
neig1_deps.fit(sign1_deps)

# %%
neig1_deps.shape

# %%
np.min(np.array(sign1_deps).flatten()), np.max(np.array(sign1_deps).flatten())

# %% [markdown]
# ### Shared DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_shared_deps.clear_all()

# Fitting sign1
sign1_shared_deps.fit(sign0_shared_deps)

# %%
sign1_shared_deps.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_shared_deps = sign1_shared_deps.diagnosis()

# Plot medium & small diagnosis plots
diag1_shared_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_shared_deps = cc_local.get_signature("neig1", "full", dataset_shared_deps)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_shared_deps.clear_all()

# Fitting neig1
neig1_shared_deps.fit(sign1_shared_deps)

# %%
neig1_shared_deps.shape

# %%
np.min(np.array(sign1_shared_deps).flatten()), np.max(np.array(sign1_shared_deps).flatten())

# %% [markdown]
# ### All Proteins

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_allprot.clear_all()

# Fitting sign1
sign1_allprot.fit(sign0_allprot)

# %%
sign1_allprot.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_allprot = sign1_allprot.diagnosis()

# Plot medium & small diagnosis plots
diag1_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_allprot = cc_local.get_signature("neig1", "full", dataset_allprot)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_allprot.clear_all()

# Fitting neig1
neig1_allprot.fit(sign1_allprot)

# %%
neig1_allprot.shape

# %%
np.min(np.array(sign1_allprot).flatten()), np.max(np.array(sign1_allprot).flatten())

# %% [markdown]
# ### All Shared Proteins

# %%
# Cleaning both full and reference datasets. This is crucial!
sign1_shared_allprot.clear_all()

# Fitting sign1
sign1_shared_allprot.fit(sign0_shared_allprot)

# %%
sign1_shared_allprot.shape

# %%
# Instantiation of diag1 (diagnosis plots)
diag1_shared_allprot = sign1_shared_allprot.diagnosis()

# Plot medium & small diagnosis plots
diag1_shared_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of neig1
neig1_shared_allprot = cc_local.get_signature("neig1", "full", dataset_shared_allprot)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1_shared_allprot.clear_all()

# Fitting neig1
neig1_shared_allprot.fit(sign1_shared_allprot)

# %%
neig1_shared_allprot.shape

# %%
np.min(np.array(sign1_shared_allprot).flatten()), np.max(np.array(sign1_shared_allprot).flatten())

# %% [markdown]
# ## sign2

# %%
# Instantiation of sign1 data structures for the new spaces: full and references
sign2_dcmoa = cc_local.signature(dataset_dcmoa, 'sign2')
sign2_dcmoa_deps = cc_local.signature(dataset_dcmoa_deps, 'sign2')
sign2_deps = cc_local.signature(dataset_deps, 'sign2')
sign2_shared_deps = cc_local.signature(dataset_shared_deps, 'sign2')
sign2_allprot = cc_local.signature(dataset_allprot, 'sign2')
sign2_shared_allprot = cc_local.signature(dataset_shared_allprot, 'sign2')

# %% [markdown]
# ### DeepCoverMoa

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_dcmoa.clear_all()

# Fit sign2 given sign1 & neig1
sign2_dcmoa.fit(sign1_dcmoa, neig1_dcmoa, oos_predictor=False)

# %%
sign2_dcmoa.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_dcmoa = sign2_dcmoa.diagnosis()

# Plot medium & small diagnosis plots
diag2_dcmoa.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_dcmoa).flatten()), np.max(np.array(sign2_dcmoa).flatten())

# %% [markdown]
# ### DeepCoverMoa DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_dcmoa_deps.clear_all()

# Fit sign2 given sign1 & neig1
sign2_dcmoa_deps.fit(sign1_dcmoa_deps, neig1_dcmoa_deps, oos_predictor=False)

# %%
sign2_dcmoa_deps.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_dcmoa_deps = sign2_dcmoa_deps.diagnosis()

# Plot medium & small diagnosis plots
diag2_dcmoa_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_dcmoa_deps).flatten()), np.max(np.array(sign2_dcmoa_deps).flatten())

# %% [markdown]
# ### DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_deps.clear_all()

# Fit sign2 given sign1 & neig1
sign2_deps.fit(sign1_deps, neig1_deps, oos_predictor=False)

# %%
sign2_deps.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_deps = sign2_deps.diagnosis()

# Plot medium & small diagnosis plots
diag2_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_deps).flatten()), np.max(np.array(sign2_deps).flatten())

# %% [markdown]
# ### Shared DEPs

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_shared_deps.clear_all()

# Fit sign2 given sign1 & neig1
sign2_shared_deps.fit(sign1_shared_deps, neig1_shared_deps, oos_predictor=False)

# %%
sign2_shared_deps.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_shared_deps = sign2_shared_deps.diagnosis()

# Plot medium & small diagnosis plots
diag2_shared_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_shared_deps).flatten()), np.max(np.array(sign2_shared_deps).flatten())

# %% [markdown]
# ### All Proteins

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_allprot.clear_all()

# Fit sign2 given sign1 & neig1
sign2_allprot.fit(sign1_allprot, neig1_allprot, oos_predictor=False)

# %%
sign2_allprot.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_allprot = sign2_allprot.diagnosis()

# Plot medium & small diagnosis plots
diag2_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_allprot).flatten()), np.max(np.array(sign2_allprot).flatten())

# %% [markdown]
# ### All Shared Proteins

# %%
# Cleaning both full and reference datasets. This is crucial!
sign2_shared_allprot.clear_all()

# Fit sign2 given sign1 & neig1
sign2_shared_allprot.fit(sign1_shared_allprot, neig1_shared_allprot, oos_predictor=False)

# %%
sign2_shared_allprot.shape

# %%
# Instantiation of diag2 (diagnosis plots)
diag2_shared_allprot = sign2_shared_allprot.diagnosis()

# Plot medium & small diagnosis plots
diag2_shared_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2_shared_allprot).flatten()), np.max(np.array(sign2_shared_allprot).flatten())

# %% [markdown]
# ## sign3

# %%
# Instantiation of sign1 data structures for the new spaces: full and references
sign3_dcmoa = cc_local.signature(dataset_dcmoa, 'sign3')
sign3_dcmoa_deps = cc_local.signature(dataset_dcmoa_deps, 'sign3')
sign3_deps = cc_local.signature(dataset_deps, 'sign3')
sign3_shared_deps = cc_local.signature(dataset_shared_deps, 'sign3')
sign3_allprot = cc_local.signature(dataset_allprot, 'sign3')
sign3_shared_allprot = cc_local.signature(dataset_shared_allprot, 'sign3')

# %%
# Get CC universe (the 25 canonical CC spaces) -- computed once, reused for all three cases below
cc_universe = []
for dat in cc_local.datasets:
    if dat.endswith('001') and dat and dat != 'D6.001':
        cc_universe.extend(cc_local.get_signature('sign2', 'full', dat).keys)
cc_universe = set(cc_universe)
print("Number of molecules in the CC universe: " + str(len(cc_universe)))

# %%
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
# ### DeepCoverMoa

# %%
# Get D6 molecules
d6_molecules_dcmoa = set(sign2_shared_allprot.keys)
 
print("Number of molecules in D6 sign2: " + str(len(d6_molecules_dcmoa)))
print("Intersection CC & D6: " + str(len(cc_universe.intersection(d6_molecules_dcmoa))))

# %%
# Cleaning both full and reference datasets. This is crucial!
sign3_dcmoa.clear_all()

# Create a list of sign2 to feed sign3 -- using the 25 CC spaces & D6
sign2_list_dcmoa = list()

# For each CC space
for ds in cc_local.coordinates:
    ds += '.001'
    sign2_list_dcmoa.append(cc_local.get_signature('sign2', 'full', ds))

# Append the new D6 space
sign2_list_dcmoa.append(cc_local.get_signature('sign2','full', dataset_dcmoa))

# In total, we now have 26 spaces
print(len(sign2_list_dcmoa))

# %%
# Fit sign3 
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_dcmoa.fit(sign2_list_dcmoa, sign2_dcmoa, sign1_dcmoa, sign2_universe=None, complete_universe="fast", sign2_coverage=None, dbconnect=False, mapping_dict=mapp)

# %%
sign3_dcmoa_arr = np.array(sign3_dcmoa)

print(sign3_dcmoa_arr.shape, np.min(sign3_dcmoa_arr), np.max(sign3_dcmoa_arr))

# %%
# Instantiation of diag3 (diagnosis plots)
diag3_dcmoa = sign3_dcmoa.diagnosis(ref_cctype='sign3')

# Plot medium & small diagnosis plots
diag3_dcmoa.canvas(size='medium', savefig=True, savefig_kwargs={'dpi': 300})
diag3_dcmoa.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %% [markdown]
# ### DEPs

# %%
# Get D6 molecules
d6_molecules_deps = set(sign2_deps.keys)
 
print("Number of molecules in D6 sign2: " + str(len(d6_molecules_deps)))
print("Intersection CC & D6: " + str(len(cc_universe.intersection(d6_molecules_deps))))

# %%
# Cleaning both full and reference datasets. This is crucial!
sign3_deps.clear_all()
 
# Create a list of sign2 to feed sign3 -- using the 25 CC spaces & D6
sign2_list_deps = list()
 
# For each CC space
for ds in cc_local.coordinates:
    ds += '.001'
    sign2_list_deps.append(cc_local.get_signature('sign2', 'full', ds))
 
# Append the new D6 space
sign2_list_deps.append(cc_local.get_signature('sign2','full', dataset_deps))
 
# In total, we now have 26 spaces
print(len(sign2_list_deps))

# %%
# Fit sign3 
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_deps.fit(sign2_list_deps, sign2_deps, sign1_deps, sign2_universe=None, complete_universe="fast", sign2_coverage=None, dbconnect=False, mapping_dict=mapp)

# %%
sign3_deps_arr = np.array(sign3_deps)
 
print(sign3_deps_arr.shape, np.min(sign3_deps_arr), np.max(sign3_deps_arr))

# %%
# Instantiation of diag3 (diagnosis plots)
diag3_deps = sign3_deps.diagnosis(ref_cctype='sign3')

# Plot medium & small diagnosis plots
diag3_deps.canvas(size='medium', savefig=True, savefig_kwargs={'dpi': 300})
diag3_deps.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %% [markdown]
# ### AllProt

# %%
# Get D6 molecules
d6_molecules_allprot = set(sign2_allprot.keys)
 
print("Number of molecules in D6 sign2: " + str(len(d6_molecules_allprot)))
print("Intersection CC & D6: " + str(len(cc_universe.intersection(d6_molecules_allprot))))

# %%
# Cleaning both full and reference datasets. This is crucial!
sign3_allprot.clear_all()
 
# Create a list of sign2 to feed sign3 -- using the 25 CC spaces & D6
sign2_list_allprot = list()
 
# For each CC space
for ds in cc_local.coordinates:
    ds += '.001'
    sign2_list_allprot.append(cc_local.get_signature('sign2', 'full', ds))
 
# Append the new D6 space
sign2_list_allprot.append(cc_local.get_signature('sign2','full', dataset_all))
 
# In total, we now have 26 spaces
print(len(sign2_list_allprot))

# %%
# Fit sign3 
# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3_allprot.fit(sign2_list_allprot, sign2_allprot, sign1_allprot, sign2_universe=None, complete_universe="fast", sign2_coverage=None, dbconnect=False, mapping_dict=mapp)

# %%
sign3_allprot_arr = np.array(sign3_allprot)
 
print(sign3_allprot_arr.shape, np.min(sign3_allprot_arr), np.max(sign3_allprot_arr))

# %%
# Instantiation of diag3 (diagnosis plots)
diag3_allprot = sign3_allprot.diagnosis(ref_cctype='sign3')
 
# Plot medium & small diagnosis plots
diag3_allprot.canvas(size='medium', savefig=True, savefig_kwargs={'dpi': 300})
diag3_allprot.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})
