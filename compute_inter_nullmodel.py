# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot DDI timelines
#
#
# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.precision', 4)
import random
from itertools import combinations
import util
from joblib import Parallel, delayed
import multiprocessing as mp
from collections import OrderedDict

np.random.seed(1)
#
# Load CSVs
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=True)

dfiu = pd.merge(dfi, dfu[['gender','age_group']], how='left', left_on='id_user', right_index=True)

dfDx = pd.read_csv('csv/drug.csv', index_col=0, encoding='utf-8')

# Set of all DrugBank Interactions
dfBDI = pd.read_csv('dictionaries/drug_interaction.csv', index_col=0, encoding='utf-8')['label'].str.split(expand=True).rename(columns={0:'i',1:'j'})
dfBDI = dfBDI.apply(lambda x: x['i']+'-'+x['j'] if (x['i']<x['j']) else x['j']+'-'+x['i'], axis='columns')
set_ddiDB = set( dfBDI.drop_duplicates(keep='first').unique() )


#print '>> dfd'
#print dfd.head()
print '>> dfc'
print dfc.head()
print '>> dfiu'
print dfiu.head()
print '>> dfu'
print dfu.head()
print '>> dfDx'
print dfDx.head()
#print '>> dfDDI'
#print dfDDI.head()



#
#
#
prob_inter = dfDx['i&j_ddi'].sum() / dfDx['i&j'].sum()
print 'Overall Prob of a CoAdmin be an Interaction:', prob_inter

#
# Null Models
#
print '-- Building Null Models'
null = list()
qt_sample_multiplier = 0.001
runs = 100
#cpu_count = int(mp.cpu_count() * 0.90)
cpu_count = 2



def null_model_user(u, n_i, n_ij, dfDx_age, prob_inter):
	
	# User has at least one co-administration
	if n_ij == 0:
		#return (u, 0.0, 0.0, 0.0)
		return (u, 0.0)
	else:
		dfDx_sample = dfDx_age.sample(n=n_i, replace=False)
		
		#p_i, p_j = zip(*random.sample(list(combinations( dfDx_sample['P(i&j_ddi)'].tolist() , 2)), n_ij))

		dfn = pd.DataFrame(
			OrderedDict([
				#('p_i',p_i),
				#('p_j',p_j)
			]), index=xrange(n_ij))
		#
		#dfn['rand'] = np.random.rand(n_ij)
		#dfn['p_i*p_j'] = dfn['p_i'] * dfn['p_j']
		#dfn['n_ij_ddi^{ind}'] = dfn.apply(lambda x: True if (np.random.rand() <= (x['p_i']*x['p_j'])) else False, axis=1)
		#dfn['n_ij_ddi^{ind}'] = dfn.apply(lambda x: True if (x['rand'] <= x['p_i*p_j']) else False, axis=1)
		#dfn['n_ij_ddi^{const}'] = dfn.apply(lambda x: True if (x['rand'] <= prob_inter) else False, axis=1)
		# {rnd}: comparison if the random pair picked is in fact a DDI.
		dfn['n_ij_ddi^{rnd}'] = [x in set_ddiDB for x in np.random.choice( [ y[0]+'-'+y[1] if y[0]<y[1] else y[1]+'-'+y[0] for y in combinations( dfDx_sample.index.tolist(), 2) ] , n_ij ) ]
		#
		d = dfn.sum(axis=0).clip(0,1).astype(int).to_dict()
		#return (u, d['n_ij_ddi^{ind}'], d['n_ij_ddi^{const}'], d['n_ij_ddi^{rnd}'])
		return (u, d['n_ij_ddi^{rnd}'])


for age_group, dft in dfu.groupby('age_group', sort=False):
	print '--- Age Group: {:s} ---'.format(age_group)

	#if age_group in ['00-04','05-09','10-14','15-19','20-24']:
	#	continue

	# A restricted set of drugs available per age
	users_in_this_age = dft.index.tolist()
	print '> users_in_this_age: {:,d}'.format( len(users_in_this_age) )
	
	dfc_age = dfc.loc[ (dfc['id_user'].isin(users_in_this_age)) , : ]
	drugs_in_this_age = np.unique(dfc_age[['db_i','db_j']])
	print '> drugs_in_this_age: {:,d}'.format( len(drugs_in_this_age) )
	dfDx_age = dfDx.loc[ (dfDx.index.isin(drugs_in_this_age)) , : ]

	qt_u_samples = int( dft.shape[0] * qt_sample_multiplier )
	print '> qt_u_samples: {:,d}'.format(qt_u_samples)
	print '-- Initiating Runs --'
	for run in np.arange(runs):
		print '> run: {:,d}'.format(run)
		
		dfts = dft.sample(n=qt_u_samples, replace=True)
		ntuple = dfts[['n_i','n_ij']].to_records()
		
		r = Parallel(n_jobs=cpu_count, prefer='threads', verbose=2)(delayed(null_model_user)(u, n_i, n_ij, dfDx_age, prob_inter) for u,n_i,n_ij in ntuple)
		#dfr = pd.DataFrame(r, columns=['u','n_ij_ddi^{ind}','n_ij_ddi^{const}','n_ij_ddi^{rnd}']).set_index('u')
		dfr = pd.DataFrame(r, columns=['u','n_ij_ddi^{rnd}']).set_index('u')
		
		# Add Gender to Results
		dfr['gender'] = dfts['gender']
		dfr['u'] = 1
		f = dfr.groupby('gender').agg('sum').to_dict()

		null.append(
			(
				age_group, run, qt_u_samples,
				f['u']['Male'],f['u']['Female'],
				#f['n_ij_ddi^{ind}']['Male'],f['n_ij_ddi^{ind}']['Female'],
				#f['n_ij_ddi^{const}']['Male'],f['n_ij_ddi^{const}']['Female'],
				f['n_ij_ddi^{rnd}']['Male'],f['n_ij_ddi^{rnd}']['Female']
			)
		)


print '>> dfN (Null Model)'
columns = [
		'age_group','run',
		'U','U^{M}','U^{F}'
		#'u^{i,M}_{ind}','u^{i,F}_{ind}',
		#'u^{i,M}_{const}','u^{i,F}_{const}',
		'U^{i,M_{rnd}','U^{i,F}_{rnd}'
	]
dfN = pd.DataFrame(null, columns=columns)

#
# Export NullModels results to .CSV
#
dfN.to_csv('csv/age_gender_null.csv.gz', compression='gzip', encoding='utf-8')




