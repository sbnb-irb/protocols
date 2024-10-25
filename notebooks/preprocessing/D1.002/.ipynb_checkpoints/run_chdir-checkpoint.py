import os
import sys
import h5py
import shutil
import tempfile
import numpy as np
import pandas as pd
import pickle
from chemicalchecker.util import Config
from chemicalchecker.util.hpc import HPC
from chemicalchecker.database import Dataset, Molrepo
from chemicalchecker.core.preprocess import Preprocess
from chemicalchecker.core.signature_data import DataSignature


CHUNK_SIZE=10  # number of tasks per single job sent to sge
dataset_code = os.path.dirname(os.path.abspath(__file__))[-6:] # D1.001 ??

## Functions
## All this functions have notebook more detailed to follow the different steps 

def parse_level(dict_dir, map_files):
    """ 
    Funtion to create the 5 needed dictionaries for running the characteristic direction 
    on LINCS data. 
    1. batch2sig --> dict that indicates to which batch correspond each signature
    2. sig2inst --> dict that indicates which replicates where used to compute each signature
    3. batch2inst --> dict that relates batch with instances
    4. batch2inst_ctr --> dict that indicates the control experimients of each batch

    dict_dir: directory to save the dictionaries
    map_files: dictionary with the paths od the files needed to parse

    """

    # files with replicate information (level 3 LINCS) and signature information (level 5 LINCS)

    # inst_info columns:  
    # ['bead_batch', 'nearest_dose', 'pert_dose', 'pert_dose_unit',
    #  'pert_idose', 'pert_time', 'pert_itime', 'pert_time_unit',
    #  'cell_mfc_name', 'pert_mfc_id', 'det_plate', 'det_well', 'rna_plate',
    #  'rna_well', 'count_mean', 'count_cv', 'qc_f_logp', 'qc_iqr', 'qc_slope',
    #  'pert_id', 'sample_id', 'pert_type', 'cell_iname', 'qc_pass',
    #  'dyn_range', 'inv_level_10', 'build_name', 'failure_mode',
    #  'project_code', 'cmap_name']

    # sig_info columns: 
    # ['bead_batch', 'nearest_dose', 'pert_dose', 'pert_dose_unit',
    #  'pert_idose', 'pert_itime', 'pert_time', 'pert_time_unit',
    #  'cell_mfc_name', 'pert_mfc_id', 'nsample', 'cc_q75', 'ss_ngene', 'tas',
    #  'pct_self_rank_q25', 'wt', 'median_recall_rank_spearman',
    #  'median_recall_rank_wtcs_50', 'median_recall_score_spearman',
    #  'median_recall_score_wtcs_50', 'batch_effect_tstat',
    #  'batch_effect_tstat_pct', 'is_hiq', 'qc_pass', 'pert_id', 'sig_id',
    #  'pert_type', 'cell_iname', 'det_wells', 'det_plates', 'distil_ids',
    #  'build_name', 'project_code', 'cmap_name', 'is_ncs_exemplar']


    instinfo_beta = os.path.join(map_files["instinfo_beta"], "instinfo_beta.txt") 
    siginfo_beta = os.path.join(map_files["siginfo_beta"], "siginfo_beta.txt") 

    sig_info = pd.read_csv(siginfo_beta, sep = "\t", dtype=str)[['sig_id, pert_id', 'pert_type', 'distil_ids']]
    sig_cp_info = sig_info[sig_info.pert_type == 'trt_cp'] # We are only interested in trt_cp instances
    del(sig_info)

    sig_id = sig_cp_info['sig_id'].unique() # list of sig_ids (720216)
    batches = set([i.split(':')[0] for i in sig_id]) # different batches in which the experiments where measured
    
    batch2sig = {}
    for i in batches:
        f=np.frompyfunc(lambda x: i in x,1,1)
        batch2sig[i] = np.array(sig_id)[np.where(f(sig_id))] # select all the sig_id with the corresponding batch in their name: batch to list of signatures 

    pickle.dump(batch2sig,open(dict_dir + 'CP_batch2sig_id.p', 'wb'))

    sig2inst = dict(zip(sig_cp_info['sig_id'], sig_cp_info['distil_ids'].str.split('|'))) # save the different intances that conform a signature: signatutr to list of instances
    pickle.dump(sig2inst,open(dict_dir + 'CP_sig2inst.p', 'wb'))

    del(sig_cp_info)

    inst_info = pd.read_csv(instinfo_beta, sep="\t", dtype=str)[['sample_id', 'pert_id', 'pert_type', 'rna_plate']]
    inst_ctr_info = inst_info[inst_info.pert_type == 'ctl_vehicle']
    inst_cp_info = inst_info[inst_info.pert_type == 'trt_cp']
    del(inst_info)

    # Create 2 dictionaries relating the batch with the instances measured in this batch. One for control instances and other for experimental
    batches2inst = {}
    inst_id = inst_cp_info['sample_id'].unique()
    for i in batches:
        batches2inst[i] = np.array(inst_id)[np.where(f(inst_id))] 

    pickle.dump(batches2inst,open(dict_dir + 'CP_batch2inst_id.p', 'wb'))

    del(inst_cp_info, inst_id)

    plate_ctr = inst_ctr_info.rna_plate.unique()
    c = set([('_').join(i.split('_')[:3]) for i in plate_ctr])
    inst_id_ctr = inst_ctr_info['sample_id'].unique()

    batch2inst_ctr = {}
    for i in c:
        batch2inst_ctr[i] = np.array(inst_id_ctr)[np.where(f(inst_id_ctr))]
    
    del(inst_ctr_info, inst_id_ctr)
    pickle.dump(batch2inst_ctr,open(dict_dir + 'CTR_batch2inst.p', 'wb'))

### All this dictionaries will be usedby the chdir_bing_cp.py script for performing the chdir to the instances ###

def select_active(pval = 2, dict_dir, root_out): 
    ''' 

    Create a list of active signatures according to the selected pval --> if a signature is active is defined by the average cosine distance between their chdir replicates
    pval = 0 --> 0.01, 1 --> 0.025, 2 --> 0.05, 3 for 0.10
    '''
    batches = set([('_').join(i.split('_')[0:3]) for i in os.listdir(root_out) if i.endswith('_chdir.h5')])
    sig2inst = pickle.load(open(dict_dir + 'CP_sig_id2distil_ids.p', 'rb'))

    significant_sig = [] 

    for i in tqdm(batches): 
        # Load sig order and pval of all the calculated CD signatures of the current batch
        with h5py.File(root_out + i + '_chdir.h5', "r") as hf: 
            sig_ids = hf['ids'][:].astype('str')
            cos_pval = hf['cos_pval'][:]
            rep = hf['rep'][:]

        # Load filtering matrix: cutoff order, number of replicates order and threshold
        with h5py.File(root_out + i + '_bck_cutoffs.h5', "r") as hf: 
            cutoff = hf['cutoffs'][:]
            replicates = hf['replicates'][:]
            values = hf['values'][:]

        if replicates.shape[0] == 1: 
            idx_sig = np.where(cos_pval< values[0][pval]) # which sig pass the 0.10 or 0.05 filter
            significant_sig.append(sig_ids[idx_sig]) # list with the identificator of the significant signatures

        else:
            for sig in range(len(sig_ids)):
                current_num = np.array(rep)[sig]
                idx_cutoff = np.where(replicates == current_num)[0][0]
                current_cutoff = values[idx_cutoff][pval]
                if cos_pval[sig] < current_cutoff:
                    significant_sig.append(sig_ids[sig]) 

    sig_pass = []
    for i in tqdm(np.array(significant_sig)): 
        if type(i) == np.ndarray: 
            for sig in i: 
                sig_pass.append(sig)
        else:
            sig_pass.append(i)
    del(significant_sig)
    return(sig_pass) ### Return a list with al sig_id significant at that level




def MODZ(sig_pass, root_out):
    '''
    For sign0, we need one signature per compound. LINCS data measured compounds in different conditions (i.e. cells, concentrations...)
    so we need to aggregate the different chdir signatures in only one per compound --> MODZ

    '''

    def open_active(sig_pass):

        '''
        Create a matrix with the chdir signatures of all that pass the significance filter
        '''
    
        batches = set([i.split(':')[0] for i in sig_pass])
        chdir = []
        sig_order = []

        for i in tqdm(batches): 
            # Load sig order and pval of all the calculated CD signatures of the current batch
            f=np.frompyfunc(lambda x: i in x,1,1)
            ids_batch = sig_pass[np.where(f(sig_pass))]
            
            with h5py.File(root_out + i + '_chdir.h5', "r") as hf: 
                sig_ids = hf['ids'][:].astype('str')
                idx = [np.where(sig_ids == i)[0][0] for i in ids_batch]
                idx.sort()
                sig_order.append(sig_ids[idx])
                chdir.append(hf['chdir'][np.array(idx)])
        
        sig_order = np.concatenate(np.array(sig_order).flatten())
        chdir = np.concatenate(np.array(chdir).flatten())

        return(sig_order, chdir)

    sig_order, chdir = open_active(sig_pass)

    # We need a list of signatures for each inchi
    # Open the molrepo to get the info for brd to inchi

    pertid_inchikey = {}
    molrepos = Molrepo.get_by_molrepo_name("lincs_2020")
    for molrepo in molrepos:
        if not molrepo.inchikey:
            continue
        pertid_inchikey[molrepo.src_id] = molrepo.inchikey 

    # get the metadata info for sig to be able to relate inchi with signatures --> to see which signatures have to be aggregated
    siginfo_beta = os.path.join(map_files["siginfo_beta"], "siginfo_beta.txt") 
    sig_info = pd.read_csv(siginfo_beta, sep = "\t", dtype=str)
    sig_info = sig_info[sig_info.sig_id.isin(sig_order)] # select the active ones
    pertid_inchikey = {key: pertid_inchikey[key] for key in sig_info.pert_id.unique() if key in pertid_inchikey.keys()} # Select in the dictionary only the drugs that pass the filter
    sig_info_group = sig_info.groupby('pert_id')
    del(sig_info)

    brd2sig = dict([(i, sig_info_group.get_group(i)['sig_id'].values) for i in pertid_inchikey.keys()])
    del(sig_info_group)

    inchi2sig= {}
    for i in tqdm(brd2sig.keys()):
        ik = pertid_inchikey[i]
        if ik not in inchi2sig:
            inchi2sig[ik] = set([])
        inchi2sig[ik].update(brd2sig[i])

    def calc_MODZ(data):
    """calculates MODZ based on the original CMAP/L1000 study"""
        if len(data)==1:
            return data.flatten()
        if len(data)==2:
            return np.mean(data,0)
        else:
            CM=scor(data.T)[0]
            fil=CM<0
            CM[fil]=0.01
            weights=np.sum(CM,1)-1
            weights=weights/np.sum(weights)
            weights=weights.reshape((-1,1))
            return np.dot(data.T,weights).reshape((-1,1)[0])

    # loop that go to all the drugs and save the different signatures 
    # with this signatures produce a consensus signatures using the modz

    consensus = []
    order_drugs = []

    for dg in tqdm(inchi2sig.keys()):
        idx_sig = np.concatenate([np.where(sig_id == i)[0] for i in inchi2sig[dg]])
        chdir_current = chdir[idx_sig]
        a = calc_MODZ(chdir_current)
        consensus.append(a)
        order_drugs.append(dg)

    return(order_drugs, consensus)


def main(args):

    args = Preprocess.get_parser().parse_args(args)

    # NS ex: fir D1.001, args will be a mapping:
    # -o : signature0full_path/raw/preprocess.h5    # --output_file --> we save the preprocess here
    # -mp: signature0full_path/raw/models           # --model_path --> here to save the dictionaries and then create a folder for the chdir
    # -m: 'fit'                                     # --method 
    ## ADD an argument for indicate the pval

    # NS added: make skip everything if the final outputfile is already present
    if os.path.exists(args.output_file) and args.method == "fit":
        main._log.info("Preprocessed file {} is already present for {}, skipping the preprocessing step.".format(args.output_file,dataset_code))
        return

    dataset = Dataset.get(dataset_code) #NS D1.001 dataset object built to queryour sql database
    path_metadata = ''
    map_files = {}  # NS: will store datasource names of D1.00X and path to the corresponding files

    # Data sources associated to this dataset are stored in map_files
    # Keys are the datasources names and values the file paths.
    # If no datasources are necessary, the list is just empty.

    try:
        for ds in dataset.datasources:
            map_files[ds.datasource_name] = ds.data_path       #Ns path is /aloy/scratch/sbnb-adm/CC/download/<datasource_name>

    except Exception as e:
        if args.method == 'fit':
            main._log.error("{}".format(e))
            sys.exit(1)
        else:
            main._log.info("Database 'dataset' cannot be accessed, will use local copies of required files if present")

    main._log.debug("Running preprocess fit method for dataset " + dataset_code + ". Saving output in " + args.output_file)
    metadatadir = os.path.join(args.models_path, "metadata") # Path to save all the dictionaries needed for the chdir job

    if os.path.exists(metadatadir) is False:
        os.makedirs(metadatadir)
    
   if args.method == 'fit':  # True

        mpath = args.models_path  # signature0full_path/raw/models/

        main._log.info("Parsing")
        parse_level(metadatadir, map_files)  #--> creates all the dictionaries needed for running chdir job

        # Creates subdirs in signature0full_path/raw/models/ --> # Path to save the chdir signatures per batch
        chdirdir = os.path.join(mpath, 'chdir')

        if os.path.exists(chdirdir) is False:
            os.makedirs(chdirdir, 0o775)

    WD = os.path.dirname(os.path.realpath(__file__))  # directory from which run.py is launched

    chdir_script = WD + "/chdir_bing_cp_2020.py"     # scripts called by run.py in the same directory. For running the chdir 

    readyfile = "conn.ready"

    config = Config()                                # reads os.environ["CC_CONFIG"]


    main._log.info("Chdir calculation")

    # CONNECTIVITY JOB 
    # OBTAIN -> 3 matrix per batch --> 
    # 'XXXX[NAME OF THE BATCH]XXX_chdir.h5' --> signature_id + average characteristic direction vectors (chdir) + number of replicates + cosine distance between their replicates
    # 'XXXX[NAME OF THE BATCH]XXX_bck_cutoffs.h5' --> cutoffs order (0.01, 0.025, 0.05, 0.1) + replicates order + values for each cutoff + distribution of random distances
    # IF NEEDED: 'XXXX[NAME OF THE BATCH]XXX_chdir_single.h5' --> sig_id + chdir for the experiments that only have one replicate so we can not calculate their significance
    # DISCLAIMER:  There are some jobs that maybe fails, possible reasons --> there is no info for the control / the metadata is not correct --> there is no inst for sig...
    # Note, connectivitydir is signature0full_path/raw/models/connectivity_fit

    if not os.path.exists(os.path.join(chdirdir, readyfile)):
        main._log.info("Getting signature files...")

        job_path = os.path.join(mpath, "job_conn")  

        if os.path.isdir(job_path):
            shutil.rmtree(job_path)
        os.mkdir(job_path)

        params = {}
        batch2sig = pickle.load(open(metadatadir + 'CP_batch2sig_id.p', 'rb'))
        num_entries = len(batch2sig.keys())

        level3_beta_trt_cp_n1805898x12328 = os.path.join(map_files["level3_beta_trt_cp_n1805898x12328"], "level3_beta_trt_cp_n1805898x12328.gctx") 
        level3_beta_ctl_n188708x12328 = os.path.join(map_files["level3_beta_ctl_n188708x12328"], "level3_beta_ctl_n188708x12328.gctx") 


        # If there are less tasks to send than the numb of tasks per job then num_jobs is just num_entries (otherwise bug since dividing by CHUNK_SIZE tells it to send 0 jobs)
        params["num_jobs"] = num_entries / CHUNK_SIZE if num_entries > CHUNK_SIZE else num_entries
        params["jobdir"] = job_path
        params["job_name"] = "CC_D1_conn"
        params["elements"] = batch2sig.keys()  
        params["memory"] = 10 # I dont know if it is enought
        
        # job command
        singularity_image = config.PATH.SINGULARITY_IMAGE
        command = "MKL_NUM_THREADS=1 singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {}".format(
            singularity_image, chdir_script, metadatadir, chdirdir, level3_beta_trt_cp_n1805898x12328, level3_beta_ctl_n188708x12328)

        # submit jobs
        cluster = HPC.from_config(config)
        cluster.submitMultiJob(command, **params)

        ### Some jobs will get an error, i dont know if it is necessary this checkpoint in this case ####
        if cluster.status() == 'error':
            main._log.error(
                "Connectivity job produced some errors. The preprocess script can't continue")
            sys.exit(1)

        if args.method == 'fit':
            with open(os.path.join(chdirdir, readyfile), "w") as f:
                f.write("")

    
    main._log.info("Selecting the significant ones")

    sig_pass = select_active(pval = 2, metadatadir, chdirdir) ### This must be an argument

    main._log.info("MODZ calculation")

    ik, chdir = MODZ(sig_pass, chdirdir)

    genes = pickle.load(open(os.path.join(chdirdir, 'genes.p'), 'rb'))

    with h5py.File(args.output_file, "w") as hf:
        hf.create_dataset("keys", data=np.array(ik, DataSignature.string_dtype()))
        hf.create_dataset("X", data=chdir)
        hf.create_dataset("features", data=np.array('genes.p', DataSignature.string_dtype())) 


    





