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
dfM = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfF = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)

#n_user = dfR['u'].sum()
n_runs = dfN['run'].max()

dfF00_89 = dfF.iloc[ 0:18, 0:4 ]
dfF90_pl = dfF.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfF = pd.concat([dfF00_89, dfF90_pl], axis=0)

print '>>dfF'
print dfF

dfM00_89 = dfM.iloc[ 0:18, 0:4 ]
dfM90_pl = dfM.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfM = pd.concat([dfM00_89, dfM90_pl], axis=0)

print '>>dfM'
print dfM

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

dfN[['u^{i,F}_{rnd}-ci_min','u^{i,F}_{rnd}-ci_max']] = dfN[['u^{i,F}_{rnd}-mean','u^{i,F}_{rnd}-std']].apply(calc_conf_interval, axis=1)
dfN[['u^{i,M}_{rnd}-ci_min','u^{i,M}_{rnd}-ci_max']] = dfN[['u^{i,M}_{rnd}-mean','u^{i,M}_{rnd}-std']].apply(calc_conf_interval, axis=1)


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
ls = '-'
ageinds = np.arange(0, dfN.shape[0])
agelabels = dfN.index.values.tolist()[:-1] + ['90+']
n_bins = 7

u_if, = ax.plot(ageinds, dfF['u^{i}'].values, color='#ff6666', marker='D', lw=lw, linestyle='-', markersize=ms, zorder=5)
u_im, = ax.plot(ageinds, dfM['u^{i}'].values, color='#b20000', marker='s', lw=lw, linestyle='-', markersize=ms, zorder=5)

ax.axvspan(9.5, 13.5, alpha=0.35, color='gray')

u_if_rnd, = ax.plot(ageinds, dfN['u^{i,F}_{rnd}-mean'].values, color='#ffcccc', marker='X', lw=lw, linestyle='dotted', markersize=(ms/2)+2, zorder=6)
u_im_rnd, = ax.plot(ageinds, dfN['u^{i,M}_{rnd}-mean'].values, color='#ffcccc', marker='P', lw=lw, linestyle='dotted', markersize=(ms/2)+2, zorder=6)

u_if_rnd_, = ax.fill(np.NaN, np.NaN, 'lightpink', edgecolor='lightgray', lw=1)
u_im_rnd_, = ax.fill(np.NaN, np.NaN, 'lightpink', edgecolor='lightgray', lw=1)

ax.fill_between(ageinds, y1=dfN['u^{i,F}_{rnd}-ci_min'].values, y2=dfN['u^{i,F}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=.5)
ax.fill_between(ageinds, y1=dfN['u^{i,M}_{rnd}-ci_min'].values, y2=dfN['u^{i,M}_{rnd}-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=.5)

# x fold Annotation
yf, ys = dfF.loc['60-64','u^{i}'].item(), dfN.loc['60-64','u^{i,F}_{rnd}-mean'].item()
diff = int(yf - ys)
diff_per = diff / yf
ymid = ( ( yf + ys )/2 )
ax.plot((12,12), (yf, ys), color='#00b200', lw=2)
ax.annotate('{:,d} ({:.0%})'.format(diff, diff_per), xy=(12, ymid), xytext=(14,ymid), fontsize=9,
	arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=0.15', facecolor='gray', edgecolor='gray'), zorder=7)

Ls = ax.legend(
	[u_if, u_im, (u_if_rnd_, u_if_rnd),(u_im_rnd_, u_im_rnd)],
	[r'$U^{\Phi,[y_1,y_2],F}$',r'$U^{\Phi,[y_1,y_2],M}$',r'$U^{\Phi,[y_1,y_2],F\star}$',r'$U^{\Phi,[y_1,y_2],M\star}$'],
	loc='upper left', handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1) #, bbox_to_anchor=(1.3, 1.125))
ax.add_artist(Ls)

ax.set_title(r'Patients with interactions by gender')
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)


ax.grid()

ax.set_xlim(-.5,len(ageinds)-.5)
ax.set_ylim(-75,1500)


print 'Export Plot File'
#plt.subplots_adjust(left=0.02, bottom=0.06, right=0.98, top=0.99, wspace=0.08, hspace=0.0)
plt.tight_layout()
plt.savefig('images/img-u-ddi-age-gender-h0.pdf', dpi=300)
plt.close()

