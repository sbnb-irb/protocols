# ---
# jupyter:
#   jupytext:
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
import os
# Specify the location of the CC config file. 
# os.environ['CC_CONFIG'] = '/path/to/your_cc_config.json'  #  e.g. chemicalchecker/setup/cc_config.json
os.environ['CC_CONFIG'] = '/aloy/home/acomajuncosa/cc_config.json'
from chemicalchecker import ChemicalChecker
ChemicalChecker.set_verbosity('DEBUG') # CRITICAL, ERROR, WARN, INFO or DEBUG
import tarfile
import wget
import os

# %%
# Modify PATH at will
PATH_TO_OUTPUT = "/aloy/home/acomajuncosa/CC_DATA"

# Create path
os.makedirs(PATH_TO_OUTPUT, exist_ok=True)


# %%
# Download data
def download_data(PATH_TO_OUTPUT):
    os.chdir(PATH_TO_OUTPUT)
    wget.download("https://chemicalchecker.com/api/db/getFile/root/sign_links.tar.gz/", out=PATH_TO_OUTPUT )

download_data(PATH_TO_OUTPUT)


# %%
# Uncompress data
def decompress_data(PATH_TO_FILE, PATH_TO_DATA):
    os.makedirs(PATH_TO_DATA, exist_ok=True)
    with tarfile.open(PATH_TO_FILE, "r:gz") as tar:
        tar.extractall(path=PATH_TO_DATA)

# Modify PATHS at will
PATH_TO_FILE = os.path.join(PATH_TO_OUTPUT, "sign_links.tar.gz")
PATH_TO_DATA = "/aloy/home/acomajuncosa/CC_DATA/DATA"  # Modify at will
decompress_data(PATH_TO_FILE, PATH_TO_DATA)
