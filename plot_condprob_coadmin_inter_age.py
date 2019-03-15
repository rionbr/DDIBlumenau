# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot Probability of Co-Admin & Interaction
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
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.precision', 4)



#
# Load CSVs
#
dfR = pd.read_csv('csv/age.csv', index_col=0, encoding='utf-8')
dfR_m = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfR_f = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')

dfR_n = pd.read_csv('csv/age_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)


# Null Computation
dfR_n['P(u)'] = dfR_n['u'] / dfR_n['u'].sum()
#dfR_n['P(c)'] = dfR_n['U^{c}'] / dfR_n['U'].sum()
dfR_n['P(i)_ind'] = dfR_n['u^{ind}'] / dfR_n['u'].sum()
dfR_n['P(i)_const'] = dfR_n['u^{const}'] / dfR_n['u'].sum()
dfR_n['P(i)_rnd'] = dfR_n['u^{rnd}'] / dfR_n['u'].sum()


print '>> dfR'
print dfR
print '>> dfR_n'
print dfR_n
print '>> dfR_m'
print dfR_m
print '>> dfR_f'
print dfR_f


dfR['P(c|y)'] = dfR['P(c)'] / dfR['P(u)']
dfR['P(i|y)'] = dfR['P(i)'] / dfR['P(u)']

dfR_m['P(c|y)'] = dfR_m['P(c)'] / dfR_m['P(u)']
dfR_m['P(i|y)'] = dfR_m['P(i)'] / dfR_m['P(u)']

dfR_f['P(c|y)'] = dfR_f['P(c)'] / dfR_f['P(u)']
dfR_f['P(i|y)'] = dfR_f['P(i)'] / dfR_f['P(u)']

dfR_n['P(i|y)_ind'] = dfR_n['P(i)_ind'] / dfR_n['P(u)']
dfR_n['P(i|y)_const'] = dfR_n['P(i)_const'] / dfR_n['P(u)']
dfR_n['P(i|y)_rnd'] = dfR_n['P(i)_rnd'] / dfR_n['P(u)']
#
# Plot
#
print '- Plotting -'
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)


print '--- Plotting ---'
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13,3))

width = 0.33
ms = 8
ms_t = 5
ms_b = 10
lw = 1.2
ageinds = np.arange(0, dfR.shape[0])
agelabels = dfR.index.values
n_bins = 7

cp_c, = ax1.plot(ageinds, dfR['P(c|y)'].values, color='orange', marker='o', lw=0, markersize=ms, zorder=5)
cp_i, = ax1.plot(ageinds, dfR['P(i|y)'].values, color='red', marker='o', lw=0, markersize=ms, zorder=5)

cp_i_ind, = ax1.plot(ageinds, dfR_n['P(i|y)_ind'].values, color='red', marker='v', lw=0, markersize=ms/2, zorder=5)
cp_i_const, = ax1.plot(ageinds, dfR_n['P(i|y)_const'].values, color='red', marker='^', lw=0, markersize=ms/2, zorder=5)
cp_i_rnd, = ax1.plot(ageinds, dfR_n['P(i|y)_rnd'].values, color='red', marker='>', lw=0, markersize=ms/2, zorder=5)


#Curve Fit
print '> Curve Fitting'

def linear(x,b0,bias):
	return (b0*x)+bias
def quadratic(x,b0,b1,bias):
	return b0*(x**2)+(b1*x)+bias
def cubic(x,b0,b1,b2,bias):
	return b0*(x**3)+b1*(x**2)+(b2*x)+bias


#ax1.legend([rrc,rri,rri_ind,rri_const,rri_rnd],[r'$RRC^{y}$',r'$RRI^{y}$',r'$H_0(RRI^y)_{ind}$',r'$H_0(RRI^y)_{const}$',r'$H_0(RRI^y)_{rnd}$'], loc=2, handletextpad=0.0)
Ps = ax1.legend(
	[cp_c,cp_i], #,cp_i_ind,cp_i_const,cp_i_rnd,],
	[r'$P(\Psi>0|y)$',r'$P(\Phi>0|y)$'],#,r'$H_0(P)_{ind}$',r'$H_0(P)_{const}$',r'$H_0(P)_{rnd}$'],
	loc=6, handletextpad=0.0)

ax1.add_artist(Ps)

Hs = ax1.legend(
	[cp_i_ind,cp_i_const,cp_i_rnd,],
	[r'$H_{ind}$',r'$H_{const}$',r'$H_{rnd}$'],
	loc=5, handletextpad=0.0, columnspacing=0, handlelength=1, ncol=3, bbox_to_anchor=(1.3, 1.125))

ax1.add_artist(Hs)

#ax1.legend([rrc,rri,rri_ind,rri_const,rri_rnd],[r'$RRC^{y}$',r'$RRI^{y}$',r'$H_0(RRI^y)_{ind}$',r'$H_0(RRI^y)_{const}$',r'$H_0(RRI^y)_{rnd}$'], loc=2, handletextpad=0.0)

cp_c_m, = ax2.plot(ageinds, dfR_m['P(c|y)'].values, color='#b27300', marker='s', markersize=ms, linestyle='dashed', zorder=5) # dark orange
cp_c_f, = ax2.plot(ageinds, dfR_f['P(c|y)'].values, color='#ffc966', marker='D', markersize=ms, linestyle='dashed', zorder=5) # light orange

#ax2.axvspan(0.5, 5.5, alpha=0.35, color='gray')

ax2.legend([cp_c_m, cp_c_f], [r'$P(\Psi>0,g=M|y)$',r'$P(\Psi>0,g=F|y)$'], loc=2, handletextpad=0.0)

cp_i_m, = ax3.plot(ageinds, dfR_m['P(i|y)'].values, color='#b20000', marker='s', markersize=ms, linestyle='dashed', zorder=5) # dark red
cp_i_f, = ax3.plot(ageinds, dfR_f['P(i|y)'].values, color='#ff6666', marker='D', markersize=ms, linestyle='dashed', zorder=5) #light red

#ax3.axvspan(6.5, 9.5, alpha=0.35, color='gray')
ax3.legend([cp_i_m, cp_i_f], [r'$P(\Phi>0,g=M|y})$',r'$P(\Phi>0,g=F|y)$'], loc=2, handletextpad=0.0)

ax1.set_title(r'$P(\Psi|y)$ and $P(\Phi|y)$')
ax1.set_xticks(ageinds)
ax1.set_xticklabels(agelabels, rotation=90)

ax1_ymax = np.round(dfR[['P(c|y)','P(i|y)']].max().max(), decimals=1)
ax1_yticks = np.linspace(0, ax1_ymax, n_bins)
ax1_yticklabels = ['%.2f'%(x) for x in ax1_yticks]
ax1.set_yticks(ax1_yticks)
ax1.set_yticklabels(ax1_yticklabels)

ax2.set_title(r'$P(\Psi,g|y)$')
ax2.set_xticks(ageinds)
ax2.set_xticklabels(agelabels, rotation=90)

ax2_ymin = np.round(min(dfR_m['P(c|y)'].min(), dfR_f['P(c|y)'].min()), decimals=1)
ax2_ymax = np.round(max(dfR_m['P(c|y)'].max(), dfR_f['P(c|y)'].max()), decimals=1)
ax2_yticks = np.linspace(0, ax2_ymax, n_bins)
ax2_yticklabels = ['%.3f'%(x) for x in ax2_yticks]
#ax2.set_yticks(ax2_yticks)
#ax2.set_yticklabels(ax2_yticklabels)

ax3.set_title(r'$P(\Phi,g|y)$')
ax3.set_xticks(ageinds)
ax3.set_xticklabels(agelabels, rotation=90)

ax3_ymax = np.round(max(dfR_m['P(i|y)'].max(), dfR_f['P(i|y)'].max()), decimals=2)
ax3_yticks = np.linspace(0, ax3_ymax, n_bins)
ax3_yticklabels = ['%.3f'%(x) for x in ax3_yticks]
ax3.set_yticks(ax3_yticks)
ax3.set_yticklabels(ax3_yticklabels)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlim(-.5,len(ageinds)-.5)
#ax1.set_ylim(0,ax1_ymax)

ax2.set_xlim(-.5,len(ageinds)-.5)
#ax2.set_ylim(ax2_ymin,ax2_ymax)

ax3.set_xlim(-.5,len(ageinds)-.5)
#ax3.set_ylim(0,ax3_ymax)


print 'Export Plot File'
#plt.subplots_adjust(left=0.02, bottom=0.06, right=0.98, top=0.99, wspace=0.08, hspace=0.0)
plt.tight_layout()
plt.savefig('images/img-condprob-coadmin-inter-age.png', dpi=300)
plt.close()



