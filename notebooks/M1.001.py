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

# %% [markdown]
# # M1.001 SPACE

# %%
import os
import wget
import tarfile

# Gathering M1 source data
# Modify PATH at will
PATH_TO_InDATA = "../data/"
link = "https://zenodo.org/records/14000612/files/M1.tar.gz?download=1"

# Create path
os.makedirs(PATH_TO_InDATA, exist_ok=True)


# %%
# Download
def download_data(PATH_TO_InDATA):
    os.chdir(PATH_TO_InDATA)
    wget.download(link, out=PATH_TO_InDATA )

download_data(PATH_TO_InDATA)


# %%
def decompress_data(PATH_TO_FILE, PATH_TO_InDATA):
    os.makedirs(PATH_TO_InDATA, exist_ok=True)
    with tarfile.open(PATH_TO_FILE, "r:gz") as tar:
        tar.extractall(path=PATH_TO_InDATA)

# Modify PATHS at will
PATH_TO_FILE = os.path.join(PATH_TO_InDATA, "M1.tar.gz")
decompress_data(PATH_TO_FILE, PATH_TO_InDATA)

# %%
# Specify the location of the CC config file. 
# os.environ['CC_CONFIG'] = '/path/to/your_cc_config.json'  #  e.g. chemicalchecker/setup/cc_config.json
os.environ['CC_CONFIG'] = '/aloy/home/acomajuncosa/cc_config.json'

from chemicalchecker import ChemicalChecker
ChemicalChecker.set_verbosity('DEBUG') # CRITICAL, ERROR, WARN, INFO or DEBUG
import numpy as np
import pandas as pd
import json
# %matplotlib inline

# %%
local_cc_dir = '../local_CC_M1'
PATH_TO_DATA = "/aloy/home/acomajuncosa/CC_DATA/DATA/"  # See Download_Data.ipynb // Procedure step 3
# PATH_TO_DATA = "/aloy/web_checker/package_cc/2021_07/sign_model_links/"
cc_local = ChemicalChecker(local_cc_dir, dbconnect=False, custom_data_path=PATH_TO_DATA)

# %% [markdown]
# ## LOAD INPUT DATA

# %%
inputFile="../data/M1/microbiota_raw.csv"
df=pd.read_csv(inputFile,index_col=0)   
print(df.shape)
df.head()

# %% [markdown]
# ## sign0 ##

# %%
# Dataset Name
dataset = 'M1.001'

# Instantiation of sign0 data structures for the new space: full and reference
sign0 = cc_local.signature(dataset, 'sign0')

# Cleaning both full and reference datasets. This is crucial!
sign0.clear_all()

# Fit sign0
sign0.fit(X=df.values, keys=list(df.index), features=list(df.columns))

# %%
sign0.shape

# %%
# Instantiation of sign0
sign0 = cc_local.signature(dataset, 'sign0')

# Instantiation of diag0 (diagnosis plots)
diag0 = sign0.diagnosis()

# Plot small diagnosis plots
diag0.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign0).flatten()), np.max(np.array(sign0).flatten())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## sign1 ##

# %%
# Dataset Name
dataset = 'M1.001'

# Instantiation of sign0
sign0 = cc_local.signature(dataset, 'sign0')

# Instantiation of sign1
sign1 = cc_local.signature(dataset, 'sign1')

# Cleaning both full and reference datasets. This is crucial!
sign1.clear_all()

# Fitting sign1
sign1.fit(sign0)

# %%
sign1.shape

# %%
# Instantiation of sign1
sign1 = cc_local.signature(dataset, 'sign1')

# Instantiation of diag1 (diagnosis plots)
diag1 = sign1.diagnosis()

# Plot medium & small diagnosis plots
diag1.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
# Instantiation of sign1
sign1 = cc_local.signature(dataset, 'sign1')

# Instantiation of neig1
neig1 = cc_local.get_signature("neig1", "full", dataset)  # It will take the reference anyway...

# Cleaning both full and reference. This is crucial!
neig1.clear_all()

# Fitting neig1
neig1.fit(sign1)

# %%
neig1.shape

# %%
np.min(np.array(sign1).flatten()), np.max(np.array(sign1).flatten())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## sign2 ##

# %%
# Dataset Name
dataset = 'M1.001'

# Get sign1
sign1 = cc_local.get_signature('sign1', 'full', dataset)

# Get neig1
neig1 = cc_local.get_signature('neig1', 'full', dataset)  # By default, all vs ref

# Instantiation of sign2
sign2 = cc_local.signature(dataset, 'sign2')

# Cleaning both full and reference datasets. This is crucial!
sign2.clear_all()

# Fit sign2 given sign1 & neig1
sign2.fit(sign1, neig1, oos_predictor=False)

# %%
sign2.shape

# %%
# Instantiation of sign2
sign2 = cc_local.signature(dataset, 'sign2')

# Instantiation of diag2 (diagnosis plots)
diag2 = sign2.diagnosis()

# Plot medium & small diagnosis plots
diag2.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
np.min(np.array(sign2).flatten()), np.max(np.array(sign2).flatten())

# %% [markdown]
# ## sign3 ##

# %%
# Dataset Name
dataset = 'M1.001'

# Get CC universe
cc_universe = []
for dat in cc_local.datasets:
    if dat != dataset and dat.endswith('001'):
        cc_universe.extend(cc_local.get_signature('sign2', 'full', dat).keys)
cc_universe = set(cc_universe)

# Get sign2
sign2 = cc_local.signature(dataset, 'sign2')

# Get M1 molecules
m1_molecules = set(sign2.keys)

print("Number of molecules in the CC universe: " + str(len(cc_universe)))
print("Number of molecules in M1 sign2: " + str(len(m1_molecules)))
print("Intersection CC & M1: " + str(len(cc_universe.intersection(m1_molecules))))

# %%
# Instantiation of sign3
sign3 = cc_local.signature(dataset, 'sign3')
sign3.clear_all()

# Create a list of sign2 to feed sign3 -- using the 25 CC spaces & M1
sign2_list = list()

# For each CC space
for ds in cc_local.coordinates:
    ds += '.001'
    sign2_list.append(cc_local.get_signature('sign2', 'full', ds))

# Append the new M1 space
sign2_list.append(cc_local.get_signature('sign2','full', dataset))

# In total, we now have 26 spaces
print(len(sign2_list))

# Get M1 sign1
sign1_self = cc_local.signature(dataset, 'sign1')

# Get M1 sign2
sign2_self = cc_local.signature(dataset, 'sign2')

# %%
# Fit sign3 

mapp = None
""" 
# Alternatively, you can provide your in-house python dictionary to map InchiKey's to InChI's (see example below)
mapp = { 
'LPXQRXLUHJKZIE-UHFFFAOYSA-N': 'InChI=1S/C4H4N6O/c5-4-6-2-1(3(11)7-4)8-10-9-2/h(H4,5,6,7,8,9,10,11)',
'BZKPWHYZMXOIDC-UHFFFAOYSA-N': 'InChI=1S/C4H6N4O3S2/c1-2(9)6-3-7-8-4(12-3)13(5,10)11/h1H3,(H2,5,10,11)(H,6,7,9)',
'XZWYZXLIPXDOLR-UHFFFAOYSA-N': 'InChI=1S/C4H11N5/c1-9(2)4(7)8-3(5)6/h1-2H3,(H5,5,6,7,8)'
} 
"""

# CAUTION: COMPUTATIONALLY DEMANDING STEP - Consider running it in an HPC cluster
sign3.fit(sign2_list, sign2_self, sign1_self, sign2_universe=None, complete_universe="fast", sign2_coverage=None, dbconnect=False, mapping_dict=mapp)

# %%
dataset = 'M1.001'

# Instantiation of sign3
sign3 = cc_local.signature(dataset, 'sign3')
sign3 = np.array(sign3)

print(sign3.shape, np.min(sign3), np.max(sign3))

# %%
# Instantiation of diag3 (diagnosis plots)
sign3 = cc_local.signature(dataset, 'sign3')
diag3 = sign3.diagnosis(ref_cctype='sign3')

# Plot medium & small diagnosis plots
diag3.canvas(size='medium', savefig=True, savefig_kwargs={'dpi': 300})
diag3.canvas(size='small', savefig=True, savefig_kwargs={'dpi': 300})

# %%
