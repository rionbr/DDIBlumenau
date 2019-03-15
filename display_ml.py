# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of ML files
#
#
from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import math
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 4)
import util
#

suffix = 'simple' # ['simple','complete','nodrug']


#
# Results
#
print '--- Results ---'
dfR = pd.read_csv('csv/ml_results_%s.csv' % (suffix), index_col=0, encoding='utf-8')
for clf, dft in dfR.groupby('clf'):
	print '> Classifier: %s' % (clf)

	dft = dft[['fold','precision','recall','f1','mcc','roc_auc','pr_auc']].copy()

	dft.loc['4'] = ['Mean'] + dft[['precision','recall','f1','mcc','roc_auc','pr_auc']].mean(axis=0).tolist()
	print dft.to_latex(index=False)

#
# Features
#
dfF = pd.read_csv('csv/ml_features_%s.csv' % (suffix), encoding='utf-8')

dfF['feature'] = dfF['feature'].replace('n_i','$\nu_i$', regex=False)
dfF['feature'] = dfF['feature'].replace('n_ij','$\Psi_{i,j}$', regex=False)
dfF['feature'] = dfF['feature'].replace('age','$y$', regex=True)
dfF['feature'] = dfF['feature'].replace('gender=Male','$g=M$', regex=True)
dfF['feature'] = dfF['feature'].replace('gender=Female','$g=F$', regex=True)

print '--- Features ---'
for clf, dft in dfF.groupby('clf'):
	print '> Classifier: %s ' % (clf)
	dft = dft.copy()
	dft['coef'] = dft['coef'].round(decimals=4)
	dft.sort_values('coef', ascending=False, inplace=True)
	n = int(math.ceil(dft.shape[0]/2))
	dft2 = pd.concat( [dft.iloc[:n , 1:].reset_index(drop=True) , dft.iloc[n: , 1:].reset_index(drop=True) ], axis=1)
	dft2.fillna('-', inplace=True)
	dft2.columns = ['feature','coef','feature','coef']
	print dft2.to_latex(index=False, escape=False)
	
