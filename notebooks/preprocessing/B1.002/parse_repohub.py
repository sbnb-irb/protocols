import h5py
import os, sys
import pandas as pd
import numpy as np
import collections

def decide(acts):
	m = np.mean(acts)
	if m > 0:
		return 1
	else:
		return -1

def string_dtype():
	if sys.version_info[0] == 2:
		import unicode
		# this works in py2 and fails in py3
		return h5py.special_dtype(vlen=unicode)
	else:
		# because str is the new unicode in py3
		return h5py.special_dtype(vlen=str)
		#return h5py.string_dtype(encoding='utf-8', length=None) 

def parse_repohub(ACTS=None, 
	repohub_file='repurposing_drugs_20200324.txt', 
	repohub_acts_file='repohub_acts.txt', 
	map_file='uniprot2name.csv'):
	
	if ACTS is None:
		ACTS = collections.defaultdict(list)
		
	# Read repohub activities from file

	dirs = {}
	with open(repohub_acts_file, 'r') as fin:
		for line in fin:
			name, act = line.strip().split('\t') 
			dirs[name] = int(act)

	# Read gene names to uniprot_ac mapping file
	
	g2uni = {}
	with open(map_file, 'r') as fin:
		fin.readline()
		for line in fin:
			uni, g = line.strip().split(',') 
			g2uni[g] = uni

	# Parse the molrepo
	
	##FIXME
	dbid_inchikey = {}
	inchikey_inchi = {}
	#molrepos = Molrepo.get_by_molrepo_name("repohub")
	#for molrepo in molrepos:
		#if not molrepo.inchikey:
			#continue
		#dbid_inchikey[molrepo.src_id] = molrepo.inchikey
		#inchikey_inchi[molrepo.inchikey] = molrepo.inchi
		
	dh_samples = pd.read_csv('repohub_molrepo.csv')
	for row_nr, row in dh_samples.iterrows():
		row = row.to_dict()
		if pd.isna(row['inchikey']):
			continue
		dbid_inchikey[row['src_id']] =  row['inchikey']
		inchikey_inchi[row['inchikey']] = row['inchikey']
	
	print('dbid_inchikey')
	print(len(dbid_inchikey))
	print('inchikey_inchi')
	print(len(inchikey_inchi))
	
	# Parse drugub
	
	dh_drugs = pd.read_csv('repurposing_drugs_20200324.txt', sep='\t', comment='!')
	dh_drugs = dh_drugs[['pert_iname', 'moa', 'target']]
	
	DB = {}
	
	for row_nr, row in dh_drugs.iterrows():
		row = row.to_dict()
		
		if  not row['pert_iname'] in dbid_inchikey:
		            continue
		inchikey = dbid_inchikey[row['pert_iname']]
		
		if pd.isna(row['target']):
			continue

		targets = collections.defaultdict(list)	
		
		if pd.isna(row['moa']):	
			moa = 'inhibitor'
		else:	
			moa = row['moa'].split(' ')[-1]
		genes = row['target'].split('|')
		for gene in genes:
			if gene in g2uni:
				uniprot_ac = g2uni[gene]
				targets[uniprot_ac] = [moa]
		
		if not targets:
			continue

		DB[inchikey] = targets

	# Save activities
	
	for inchikey, targs in DB.items():
		for uniprot_ac, actions in targs.items():
			if (inchikey, uniprot_ac, inchikey_inchi[inchikey]) in ACTS:
				continue
			d = []
			for action in actions:
				if action in dirs:
					d += [dirs[action]]
			if not d:
				continue
			act = decide(d)
			ACTS[(inchikey, uniprot_ac, inchikey_inchi[inchikey])] = act

	return ACTS

if __name__ == '__main__':

	ACTS = parse_repohub()
	print('ACTS')
	print(len(ACTS))
	
	feats = []
	inchikey_raw = list()
	for k, v in ACTS.items():
		inchikey_raw.append((k[0], k[1] + "(" + str(v) + ")"))
		feats += [k[1]]
	
	print('Unique features %d' % len(set(feats)))

	output_file = 'repohub_pairs.h5'
	with h5py.File(output_file, "w") as hf:
		hf.create_dataset("pairs", data=np.array(inchikey_raw, dtype=string_dtype()))
	
	with h5py.File(output_file, "r") as hf:
		cc = hf["pairs"][:]
		print('cc')
		print(cc.shape)
		print(cc)

