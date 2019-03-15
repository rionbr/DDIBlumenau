# coding=utf-8
# Author: Rion B Correia
# Date: Nov 29, 2018
#
# Description: 
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
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
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
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_inter_null_models.py`)

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

dfR['RI^{y}'] = (dfR['u^{i}'] ) / (dfR['u^{c}'])

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
# Calculate RI
#
dfN['RI^{y}_{rnd}'] = dfN['u^{i}_{rnd}-mean'] / dfR['u^{c}']

#
# Confidence Interval
#
dfN[['u^{i}_{rnd}-ci_min','u^{i}_{rnd}-ci_max']] = dfN[['u^{i}_{rnd}-mean','u^{i}_{rnd}-std']].apply(calc_conf_interval, axis=1)
dfN['RI^{y}_{rnd}-ci_min'] = dfN['u^{i}_{rnd}-ci_min'] / dfR['u^{c}']
dfN['RI^{y}_{rnd}-ci_max'] = dfN['u^{i}_{rnd}-ci_max'] / dfR['u^{c}']


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


ri, = ax.plot(ageinds, dfR['RI^{y}'].values, color='red', marker='o', lw=lw, linestyle='', markersize=ms, zorder=5)

ax.axvspan(9.5, 13.5, alpha=0.35, color='gray')

ri_rnd, = ax.plot(ageinds, dfN['RI^{y}_{rnd}'].values, color='red', marker='*', lw=lw, linestyle=ls, markersize=ms, zorder=6)
ri_rnd_, = ax.fill(np.NaN, np.NaN, 'lightpink', edgecolor='lightgray', lw=1)

ax.fill_between(ageinds, y1=dfN['RI^{y}_{rnd}-ci_min'].values, y2=dfN['RI^{y}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=1)



#
#Curve Fitting
#
print '--- Curve Fitting ---'
y_ri = dfR['RI^{y}'].values
x = np.arange(len(y_ri))
# x_ is used to plot the curve after
x_ = np.linspace(x[0],x[-1],len(x)*10)

# RI Linear Model
print '> RI Linear Model'
Xc = sm.add_constant( x )
ri_l_model = sm.OLS(y_ri, Xc)
ri_l_model_result = ri_l_model.fit()
#print ri_l_model_result.summary()

# RI Cubic Model
print '> RI CUBIC'
Xc = sm.add_constant( np.column_stack( [x**3, x**2, x] ) )
ri_c_model = sm.OLS(y_ri, Xc)
ri_c_model_result = ri_c_model.fit()
print ri_c_model_result.summary()
Xc_ = sm.add_constant( np.column_stack( [x_**3, x_**2, x_] ) )
y_ri_ = np.dot(Xc_, ri_c_model_result.params)
ri_f_cubic = ax.plot(x_, y_ri_, color='red', lw=2, zorder=3)
ri_c_model_R2 = ri_c_model_result.rsquared_adj

# ANOVA
#anova_result = anova_lm(ri_l_model_result, ri_c_model_result)
#print anova_result

ax.text(x=0.97, y=0.03, s=r'$R^2={:.3f}$'.format(ri_c_model_R2), ha='right',va='bottom', transform=ax.transAxes)


Ls = ax.legend(
	[ri, (ri_rnd_, ri_rnd) ],
	[r'$RI^{[y_1,y_2]}$' , r'$RI^{[y_1,y_2]\star} [H^{rnd}_0]$' ],
	loc='upper left', handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1) #, bbox_to_anchor=(1.3, 1.125))

ax.add_artist(Ls)

ax.set_title(r'$RI^{[y_1,y_2]}$')
#ax.text(0, 1.09, r'$\frac{ P(\Phi^{u}>0|u \in U^{[y_1,y_2]}) }{ P(\Psi^{u}>0|u \in U^{[y_1,y_2]}) }$', transform=ax.transAxes, fontsize=14)
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)
ax.grid()
ax.set_xlim(-.5,len(ageinds)-.5)
ax.set_ylim(-0.03,0.38)


print 'Export Plot File'
#plt.subplots_adjust(left=0.02, bottom=0.06, right=0.98, top=0.99, wspace=0.08, hspace=0.0)
plt.tight_layout()
plt.savefig('images/img-ri-age-h0.pdf', dpi=300)
plt.close()



