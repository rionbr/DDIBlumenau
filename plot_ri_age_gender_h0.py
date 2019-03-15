# coding=utf-8
# Author: Rion B Correia
# Date: Nov 15, 2018
#
# Description: Plot DDI timelines
#
#
# coding=utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
#from scipy.optimize import curve_fit
from scipy.stats import chisquare, ks_2samp
#from scipy.stats.distributions import t, f
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from collections import OrderedDict
#from lmfit import Model
#from scipy import stats
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.precision', 4)
import scipy.stats as stats
import math


def linear(x,b0,bias):
	return (b0*x)+bias
def quadratic(x,b0,b1,bias):
	return b0*(x**2)+(b1*x)+bias
def cubic(x,b0,b1,b2,bias):
	return b0*(x**3)+b1*(x**2)+(b2*x)+bias

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
dfM = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfF = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)

n_user = dfR['u'].sum()
n_runs = dfN['run'].max()
n_user_female = dfF['u'].sum()
n_user_male = dfM['u'].sum()
print 'Number of users: {:d}'.format(n_user)


dfF00_89 = dfF.iloc[ 0:18, 0:4 ]
dfF90_pl = dfF.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfF = pd.concat([dfF00_89, dfF90_pl], axis=0)

dfM00_89 = dfM.iloc[ 0:18, 0:4 ]
dfM90_pl = dfM.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfM = pd.concat([dfM00_89, dfM90_pl], axis=0)

dfN['u^{i}_{rnd}'] = dfN['u^{i,F}_{rnd}'] + dfN['u^{i,M}_{rnd}']
dfN = dfN.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	('u^{i}_{rnd}',['mean','std']),
	('u^{i,F}_{rnd}',['mean','std']),
	('u^{i,M}_{rnd}',['mean','std'])
	]))
dfN.columns = ['-'.join(col).strip() for col in dfN.columns.values]
dfN00_89 = dfN.iloc[ 0:18, : ]
dfN90_pl = dfN.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfN = pd.concat([dfN00_89, dfN90_pl], axis=0)
# Adding number of Male & Femals - Only true if random sample is 100%
dfN['u'] = dfM['u'] + dfF['u']
dfN['u^{M}'] = dfM['u']
dfN['u^{F}'] = dfF['u']


dfF['RI^{y}'] = dfF['u^{i}'] / dfF['u^{c}']
dfM['RI^{y}'] = dfM['u^{i}'] / dfM['u^{c}']

dfN['RI^{y,F}_{rnd}'] = dfN['u^{i,F}_{rnd}-mean'] / dfF['u^{c}']
dfN['RI^{y,M}_{rnd}'] = dfN['u^{i,M}_{rnd}-mean'] / dfM['u^{c}']

# Confidence Interval
dfN[['u^{i,F}_{rnd}-ci_min','u^{i,F}_{rnd}-ci_max']] = dfN[['u^{i,F}_{rnd}-mean','u^{i,F}_{rnd}-std']].apply(calc_conf_interval, axis=1)
dfN[['u^{i,M}_{rnd}-ci_min','u^{i,M}_{rnd}-ci_max']] = dfN[['u^{i,M}_{rnd}-mean','u^{i,M}_{rnd}-std']].apply(calc_conf_interval, axis=1)
dfN['RI^{y,F}_{rnd}-ci_min'] = dfN['u^{i,F}_{rnd}-ci_min'] / dfF['u^{c}']
dfN['RI^{y,F}_{rnd}-ci_max'] = dfN['u^{i,F}_{rnd}-ci_max'] / dfF['u^{c}']
dfN['RI^{y,M}_{rnd}-ci_min'] = dfN['u^{i,M}_{rnd}-ci_min'] / dfM['u^{c}']
dfN['RI^{y,M}_{rnd}-ci_max'] = dfN['u^{i,M}_{rnd}-ci_max'] / dfM['u^{c}']

print '>> dfM'
print dfM
print '>> dfF'
print dfF
print '>> dfN (null model)'
print dfN

print '>> Maximal Difference RI^{y,F} - RI^{y,M}'
print dfF['RI^{y}'] - dfM['RI^{y}']
print '>> Maximal Difference RI^{y,F} - RI^{y,F}_{rnd}'
print dfF['RI^{y}'] - dfN['RI^{y,F}_{rnd}']

#
#
#
print '--- Plotting ---'
fig, ax = plt.subplots(figsize=(4.3,3), nrows=1, ncols=1)
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)


width = 0.33
ms = 8
ms_t = 5
ms_b = 10
lw = 2
ls = 'dashed'
ageinds = np.arange(0, dfF.shape[0])
agelabels = dfF.index.values

ri_f, = ax.plot(ageinds, dfF['RI^{y}'].values, color='#ff6666', marker='D', markersize=ms, lw=lw, ls=ls, zorder=5)
ri_m, = ax.plot(ageinds, dfM['RI^{y}'].values, color='#b20000', marker='s', markersize=ms, lw=lw, ls=ls, zorder=5)

ax.axvspan(9.5, 13.5, alpha=0.35, color='gray')

#ri_f_rnd, = ax.plot(ageinds, dfN['RI^{y,F}_{rnd}'].values, color='#ff6666', marker='X', markersize=(ms/2)+2, linestyle='', zorder=6) #light red
#ri_m_rnd, = ax.plot(ageinds, dfN['RI^{y,M}_{rnd}'].values, color='#b20000', marker='P', markersize=(ms/2)+2, linestyle='', zorder=6) # dark red

#ax.fill_between(ageinds, y1=dfN['RI^{y,F}_{rnd}-ci_min'].values, y2=dfN['RI^{y,F}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=.5)
#ax.fill_between(ageinds, y1=dfN['RI^{y,M}_{rnd}-ci_min'].values, y2=dfN['RI^{y,M}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=.5)


# x fold Annotation
ym, yf = dfM.loc['60-64','RI^{y}'].item(), dfF.loc['60-64','RI^{y}'].item()
diff = yf - ym
ymid = ( ( yf + ym )/2 )
ax.plot((12,12), (yf, ym), color='#00b200', lw=2)
ax.annotate('{:.2%}'.format(diff), xy=(12, ymid), xytext=(14,ymid-0.06), fontsize=10,
	arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=-0.15', facecolor='gray', edgecolor='gray'))


ax.legend([ri_f,ri_m],[r'$RI^{[y_1,y_2],F}$',r'$RI^{[y_1,y_2],M}$'],
	loc=2, handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1)

ax.set_title(r'$RI^{[y_1,y_2],g}$')
#ax.text(0, 1.09, r'$\frac{ P(\Phi^{u}>0|u \in U^{[y_1,y_2],g}) }{ P(\Psi^{u}>0|u \in U^{[y_1,y_2],g}) }$', transform=ax.transAxes, fontsize=14)
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)
ax.grid()
ax.set_xlim(-.6,len(ageinds)-0.4)
ax.set_ylim(-.02,0.41)

print 'Export Plot File'
#plt.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.92, wspace=0.20, hspace=0.20)
plt.tight_layout()
plt.savefig('images/img-ri-age-gender-h0.pdf', dpi=300)
plt.close()



