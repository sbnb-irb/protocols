import os
import sys
import pickle
import time
import h5py
import random

# Basic Utilities 
import pandas as pd
import numpy as np
from tqdm import tqdm

# Compute distances
from scipy.spatial.distance import pdist

# Characteristic Direction package
from geode import *
# Cmap method to open the gctx files
from cmapPy.pandasGEXpress.parse import parse



batch_id = sys.argv[1]  # Current batch
root_LINCS = sys.argv[2]  # path to computed dictionaries
root_out = sys.argv[3] # path where chdir signatures are saved
level3_beta_trt_cp_n1805898x12328 = sys.argv[4] # path to cp data
level3_beta_ctl_n188708x12328 = sys.argv[5] # path to control data
print("Processing batch_id: ", batch_id)


"""
root_LINCS = '/aloy/home/epareja/TFM/Script/LINCS/6_Rebuttal_Protocols/models_out/metadata'
root_out = '/aloy/home/epareja/TFM/Script/LINCS/6_Rebuttal_Protocols/models_out/chdir'
level3_beta_trt_cp_n1805898x12328 = '/aloy/home/epareja/TFM/Script/LINCS/6_Rebuttal_Protocols/infiles/level3_beta_trt_cp_n1805898x12328.gctx'
level3_beta_ctl_n188708x12328 = '/aloy/home/epareja/TFM/Script/LINCS/6_Rebuttal_Protocols/infiles/level3_beta_ctl_n188708x12328.gctx'
"""


batch2sig = pickle.load( open( os.path.join(root_LINCS, 'CP_batch2sig_id.p'), 'rb' ) )
sig2inst = pickle.load( open( os.path.join(root_LINCS, 'CP_sig_id2distil_ids.p'), 'rb' ) )
batch2inst = pickle.load( open( os.path.join(root_LINCS, 'CP_batch2inst_id.p'), 'rb' ) )
ctr_batch2inst = pickle.load( open( os.path.join(root_LINCS, 'CTR_batch2inst.p'), 'rb' ) )

def CD_replicate(selected_samples, controls_selected): 
    '''
    This function takes the instances and select them depending on the plate. Then go to the row data and obtain the expression
    of this instances. 
    Return the input to the characteristic direction funcion: 
        - exp: matrix with the expression values for all the instances
        - sample_class: list with the order of the samples in the matrix (2 for experiment and 1 for control)
        - genes
    '''
    
    plates = set([('_').join(i.split('_')[:4]) for i in selected_samples])
    CD_rep = []
    
    for i in plates:
        f=np.frompyfunc(lambda x: i in x,1,1)
        exp_id = []
        sample_plate = np.array(selected_samples)[np.where(f(np.array(selected_samples)))] # select only the samples of this plate 
        control_plate = np.array(controls_selected)[np.where(f(np.array(controls_selected)))] # select only the control of this plate
        exp_id.append(sample_plate)
        exp_id.append(control_plate)
        if sample_plate.shape[0] == control_plate.shape[0]: 
            exp_id = list(np.array(exp_id, dtype=object).flatten())
        else:        
            exp_id = list(np.concatenate(np.array(exp_id, dtype=object).flatten()))
        current_exp = np.array(profiles[exp_id])# select the GEx
        expetiment_ind = np.repeat(2, sample_plate.shape[0])
        control_ind = np.repeat(1, control_plate.shape[0])
        sample_class = (expetiment_ind, control_ind)
        sample_class = np.concatenate(sample_class) # create the sample class vector 
        genes = list(batch_exp_data.index)
        CD_rep.append(chdir(current_exp, sample_class, genes, sort = False)) # you have to indicate the GEx data, 
                                                                            # a vector with 2 and 1 indicanting either an experiment or a control sample, genes order --> see chdir documentation
    
    return(CD_rep)
    
# Select all the control and experimental instances in this batch
try:
    ctr = ctr_batch2inst[batch_id]
    exp_replicates = batch2inst[batch_id]
    batch_exp_data = parse(level3_beta_trt_cp_n1805898x12328, cid = exp_replicates).data_df
    batch_ctr_data = parse(level3_beta_ctl_n188708x12328, cid = ctr).data_df
    profiles = batch_exp_data.merge(batch_ctr_data, right_index=True, left_index=True)
    genes = list(profiles.index)

    sig2number = {}
    experiment = []
    rep = []
    no_rep = []
    experiment_no_rep = []
    # For each signature in the current batch --> compute the chdir for each of its replicates
    for sig in tqdm(batch2sig[batch_id]): 
        exp_names = sig2inst[sig]
        try:
            CD_rep = CD_replicate(exp_names, ctr)
             # When there is no replicates --> we save the Chdir for this replicate but we can not calculate the average either the cosine distance
            if len(CD_rep) == 1: 
                no_rep.append(np.array([chdir[0] for chdir in CD_rep[0]]))
                experiment_no_rep.append(sig)
                continue    
            
            # Save the replicates and the number of replicates for this experiment
            for r in CD_rep: 
                rep.append(np.array([a_tuple[0] for a_tuple in r])) # list chdir replicates for each signature
                experiment.append(sig) # save the order of sign
                
            sig2number[sig] = len(CD_rep)   # relate the sig_id with the number of replicates
        except:
            pass
      
    # To assess the significance of the experiement. 
    # Average cosine distance between the replicates compared with the background distribution of the batch        

    exp2dist = {}
    exp2cut = {}
    significative = []
    CD_avgs = []
    permuCount = 10000
    bck_cutoffs = [100, 250, 500, 1000]
    number_of_replicates = []

    if sig2number: 
        for i in tqdm(sig2number):

            distVal = []
            currentRepCount = sig2number[i]

            if currentRepCount not in list(exp2dist.keys()): # Only calculate the background distribution once per number of replicates
                for j in range(permuCount):
                    permu = random.sample(range(np.array(rep).shape[0]), currentRepCount) # calculate distances between random replicates present on the batch
                    sample = np.array(rep)[permu]
                    distVal.append(np.mean(pdist(sample, 'cosine')))

                exp2dist[currentRepCount] = distVal
                exp2dist[currentRepCount].sort()
                exp2cut[currentRepCount] = np.array(exp2dist[currentRepCount])[bck_cutoffs] # save the distribution of random cosines distances and the cutoffs

            current_sig = np.array(rep)[np.where(np.array(experiment) == i)[0]] # select the acctual replicates and calculate the cosine distance
            exp_cos = np.mean(pdist(current_sig, 'cosine'))
            number_of_replicates.append(current_sig.shape[0])

            significative.append([i,exp_cos])
            CD_avgs.append(np.mean(current_sig, axis=0)/np.linalg.norm(np.mean(current_sig, axis=0)))  


    if significative: 

        with h5py.File( os.path.join( root_out, '%s_chdir.h5'%batch_id ) , "w") as o:
            o.create_dataset('ids', data=np.array([x[0] for x in significative], dtype='S'))
            o.create_dataset('cos_pval', data=np.array([x[1] for x in significative]))
            o.create_dataset('chdir', data=np.asarray(CD_avgs))
            o.create_dataset('rep', data=np.asarray(number_of_replicates))

        with h5py.File( os.path.join( root_out, '%s_bck_cutoffs.h5'%batch_id ), "w") as o:
            o.create_dataset('cutoffs', data=np.round(np.array(bck_cutoffs)/permuCount,4))
            rep2v = list(exp2cut.items())
            rep2d = list(exp2dist.items())
            o.create_dataset('replicates', data= np.asarray([x[0] for x in rep2v]))
            o.create_dataset('values', data= np.asarray([x[1] for x in rep2v]))
            o.create_dataset('distribution', data = np.asarray([x[1] for x in rep2d]))


    if no_rep: 
        with h5py.File( os.path.join( root_out, '%s_chdir_single.h5'%batch_id ), "w") as o:
            o.create_dataset('ids', data=np.array(experiment_no_rep, dtype='S'))
            o.create_dataset('chdir', data=np.asarray(no_rep))

    genes_file = os.path.join(root_out, 'genes.p')

    if not os.path.exists(genes_file): 
        pickle.dump(genes,open(genes_file, 'wb'))
except:
    pass

