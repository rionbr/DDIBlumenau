# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Compute Individual Drug Probability to be used by NullModels
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
import util
from collections import OrderedDict
#
# Load CSVs
#
dfd = pd.read_csv('results/dd_drugs.csv.gz', header=0, encoding='utf-8', nrows=None)

dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=True)
n_users = dfu.shape[0]

print '>> dfd'
print dfd.head()
print '>> dfc'
print dfc.head()
print '>> n_users: {:d}'.format(n_users)


#
# Drugs
#
dfdg = dfd.groupby('DB_label').agg(
	OrderedDict([
		('en_i','first'),
		('id_user',pd.Series.nunique),
		('n_disp','sum'),
	])).reset_index().rename(columns={'id_user':'u','n_disp':'a','DB_label':'db_i'}).set_index('db_i')

dfdg['P(u)'] = dfdg['u'] / n_users
#
# Drugs with other drugs
#
d = []
for db_i in np.unique(dfc[['db_i','db_j']]):
	dft = dfc.loc[
		(
			(
				(dfc['db_i']==db_i) | (dfc['db_j']==db_i)
			) & (
				(dfc['db_i']!=dfc['db_j'])
			)
		) , : ]
	#
	dfti = dft.loc[ dft['inter']==1 , : ]
	#
	i_j = dft.shape[0]
	i_j_ddi = dfti.shape[0]
	u_i_j = len(dft['id_user'].unique())
	u_i_j_ddi = len(dfti['id_user'].unique())

	d.append( (db_i,
		i_j, # n_ij_male, n_ij_female,
		i_j_ddi, # n_ij_ddi_male, n_ij_ddi_female) )
		u_i_j,
		u_i_j_ddi
	))


dfDx = pd.DataFrame(d, columns=['db_i','i&j','i&j_ddi','u_i&j','u_i&j_ddi']).set_index('db_i')


dfDx['P(i&j_ddi)'] = dfDx['i&j_ddi'] / dfDx['i&j']
dfDx['P(i&j_ddi)'].fillna(0, inplace=True)

dfDx['P(u_i&j_ddi)'] = dfDx['u_i&j_ddi'] / dfDx['u_i&j']
dfDx['P(u_i&j_ddi)'].fillna(0, inplace=True)

print dfDx.head()

# Merge
dfD = dfdg.merge(dfDx, how='inner', left_index=True, right_index=True)
print dfD.head()

dfD.to_csv('csv/drug.csv', encoding='utf-8')

