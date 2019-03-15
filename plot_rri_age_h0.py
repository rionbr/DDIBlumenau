# coding=utf-8
# Author: Rion B Correia
# Date: Jan 29, 2018
#
# Description: Plot results from Null Models
#
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.precision', 4)
from collections import OrderedDict
import scipy.stats as stats
import math


def calc_conf_interval(r, **kwargs):
	df = n_runs-1
	mean = r.iloc[0]
	std = r.iloc[1]
	sigma = std/math.sqrt(n_runs)
	(ci_min,ci_max) = stats.t.interval(alpha=0.95, df=n_runs-1, loc=mean, scale=sigma)
	return pd.Series([ci_min, ci_max], index=['ci_min', 'ci_max'])


#
# Load CSVs
#
dfR = pd.read_csv('csv/age.csv', index_col=0, encoding='utf-8')
#dfR = pd.read_csv('csv/age_short.csv', index_col=0, encoding='utf-8')
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)

# Sum Males + Females
dfN['u^{i}_{rnd}'] = dfN['u^{i,F}_{rnd}'] + dfN['u^{i,M}_{rnd}']

n_user = dfR['u'].sum() # This is only valid if the sample of users is 100%
n_runs = dfN['run'].max()

# Group all the Runs in the NullModel
dfN = dfN.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	('u^{i}_{rnd}',['mean','std']),
]))
dfN.columns = ['-'.join(col).strip() for col in dfN.columns.values]


#dfR00_89 = dfRiloc[ 0:4 , : ].sum(axis=0).to_frame(name='00-19').T
dfR00_89 = dfR.iloc[ 0:18, : ]
dfR90_pl = dfR.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfR = pd.concat([dfR00_89, dfR90_pl], axis=0)

#dfR['RRC^{y}'] = (dfR['u^{c}'] / dfR['u']) / (dfR.loc['00-04':'15-19','u^{c}'].sum() / dfR.loc['00-04':'15-19','u'].sum())
#dfR['RRI^{y}'] = (dfR['u^{i}'] / dfR['u']) / (dfR['u^{c}'] / dfR['u'])
dfR['RRI^{y}'] = (dfR['u^{i}'] ) / (dfR['u^{c}'])

print '>> dfR'
print dfR

#
# Null Models Computation - Short
#
# Compute the dfN
dfN00_89 = dfN.iloc[ 0:18, : ]
dfN90_pl = dfN.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfN = pd.concat([dfN00_89, dfN90_pl], axis=0)
print dfN
#
# Calculate RRI from H_0 models
#
dfN['RRI^{y}_{rnd}'] = (dfN['u^{i}_{rnd}-mean'] / dfN['u-mean']) / (dfR['u^{c}'] / dfR['u'])

#
# Calculare RRI from real data
#
dfN['RRI^{y}_{rnd}_real'] = (dfN['u^{i}_{rnd}-mean'] / dfN['u-mean']) / (dfR['u^{c}'] / dfR['u'])

#
# Confidence Interval
#
dfN[['u^{i}_{rnd}-ci_min','u^{i}_{rnd}-ci_max']] = dfN[['u^{i}_{rnd}-mean','u^{i}_{rnd}-std']].apply(calc_conf_interval, axis=1)
dfN['RRI^{y}_{rnd}-ci_min'] = (dfN['u^{i}_{rnd}-ci_min'] / dfN['u-mean']) / (dfR['u^{c}'] / dfR['u'])
dfN['RRI^{y}_{rnd}-ci_max'] = (dfN['u^{i}_{rnd}-ci_max'] / dfN['u-mean']) / (dfR['u^{c}'] / dfR['u'])


print '>> dfN (Null Model)'
print dfN.head()
#
# Plot
#
print '- Plotting -'
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)


print '--- Plotting ---'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.3,3))

width = 0.33
ms = 8
ms_t = 5
ms_b = 10
lw = 2
ls = ''
ageinds = np.arange(0, dfR.shape[0])
agelabels = dfR.index.values
n_bins = 7


rri, = ax.plot(ageinds, dfR['RRI^{y}'].values, color='red', marker='o', lw=lw, linestyle='-', markersize=ms, zorder=5)

#rri_ind, = ax.plot(ageinds, dfR_n['RRI^{y}_ind'].values, color='tomato', marker='*', lw=lw, linestyle=ls, markersize=(ms/2)+2, zorder=8)
#rri_const, = ax.plot(ageinds, dfR_n['RRI^{y}_const'].values, color='maroon', marker='^', lw=lw, linestyle=ls, markersize=ms/2, zorder=7)
rri_rnd, = ax.plot(ageinds, dfN['RRI^{y}_{rnd}'].values, color='red', marker='*', lw=lw, linestyle=ls, markersize=ms, zorder=6)

#rri_ind_, = ax.fill(np.NaN, np.NaN, 'lightblue', edgecolor='lightgray', lw=.5)
#rri_const_, = ax.fill(np.NaN, np.NaN, 'lightgreen', edgecolor='lightgray', lw=.5)
rri_rnd_, = ax.fill(np.NaN, np.NaN, 'lightpink', edgecolor='lightgray', lw=1)

#ax.fill_between(ageinds, y1=dfR_n['RRI^{y}_ind-ci_min'].values, y2=dfR_n['RRI^{y}_ind-ci_max'].values, color='lightblue', edgecolor='lightgray', lw=.5)
#ax.fill_between(ageinds, y1=dfR_n['RRI^{y}_const-ci_min'].values, y2=dfR_n['RRI^{y}_const-ci_max'].values, color='lightgreen', edgecolor='lightgray', lw=.5)
ax.fill_between(ageinds, y1=dfN['RRI^{y}_{rnd}-ci_min'].values, y2=dfN['RRI^{y}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=1)

#rri_ind, = ax.plot(ageinds, dfRs_n['RRI^{y}_ind'].values, color='red', marker='*', lw=0, markersize=(ms/2)+2, zorder=8)
#rri_const, = ax.plot(ageinds, dfRs_n['RRI^{y}_const'].values, color='red', marker='^', lw=0, markersize=ms/2, zorder=7)
#rri_rnd, = ax.plot(ageinds, dfRs_n['RRI^{y}_rnd'].values, color='red', marker='D', lw=0, markersize=ms/2, zorder=6)

"""
Ls = ax.legend(
	[rri, (rri_ind_,rri_ind), (rri_const_, rri_const), (rri_rnd_, rri_rnd) ],
	[r'$RRI^{y}$', r'$H^{ind}_0$' , r'$H^{const}_0$' , r'$H^{rnd}_0$' ],
	loc='upper left', handletextpad=0.5, columnspacing=0, handlelength=1, ncol=1) #, bbox_to_anchor=(1.3, 1.125))
"""
Ls = ax.legend(
	[rri, (rri_rnd_, rri_rnd) ],
	[r'$RRI^{y}$' , r'$RRI^{y\star} [H^{rnd}_0]$' ],
	loc='upper left', handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1) #, bbox_to_anchor=(1.3, 1.125))

ax.add_artist(Ls)

ax.set_title(r'$RRI^{y}$')
ax.text(0, 1.09, r'$\frac{ P(\Phi^{u}>0|u \in U^{[y_1,y_2]}) }{ P(\Psi^{u}>0|u \in U^{[y_1,y_2]}) }$', transform=ax.transAxes, fontsize=14)
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)
ax.grid()
ax.set_xlim(-.5,len(ageinds)-.5)
ax.set_ylim(-0.03,0.38)


print 'Export Plot File'
#plt.subplots_adjust(left=0.02, bottom=0.06, right=0.98, top=0.99, wspace=0.08, hspace=0.0)
plt.tight_layout()
plt.savefig('images/img-rri-age-h0.pdf', dpi=300)
plt.close()



