# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of age of users on interactin drugs
#
#
# coding=utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import string
import numpy as np
from scipy.stats import norm, ttest_ind
import statsmodels.api as sm
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import util
import numpy as np
import math


ints = [0] + list(np.arange(20, 85, 5))
strs = ['%02d-%02d' % (x, x1-1) for x, x1 in zip(ints[:-1], ints[1:]) ] + ['>80']
def apply_age_group(x):
	g = np.digitize([x], ints, right=False)[0]
	return strs[g-1]

#
# Load Interaction / Users
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(apply_age_group=apply_age_group, cat_age_groups=strs)


# Only those with dispensations
#dfusgD = dfusg.loc[ (dfusg['qt_drugs']>0) , : ]
dfD = dfu.groupby(['gender','age_group']).agg({'n_i':['mean','std']})
dfD.columns = dfD.columns.droplevel(level=0)
dfDm = dfD.loc['Male']
dfDf = dfD.loc['Female']

# Only those with co-administrations
dfC = dfu.loc[ (dfu['n_ij']>0) , : ]
dfC = dfC.groupby(['gender','age_group']).agg({'n_ij':['mean','std']})
dfC.columns = dfC.columns.droplevel(level=0)
dfCm = dfC.loc['Male']
dfCf = dfC.loc['Female']

# Only those with interactions
dfI = dfu.loc[ (dfu['n_ij_ddi']>0) , : ]
dfI = dfI.groupby(['gender','age_group']).agg({'n_ij_ddi':['mean','std']})
dfI.columns = dfI.columns.droplevel(level=0)
dfIm = dfI.loc['Male']
dfIf = dfI.loc['Female']

#
# Plotting
#
print '- Plotting -'
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11,3))
plt.rc('font', size=12)
plt.rc('legend', fontsize=9)
plt.rc('legend', numpoints=1)

print 'Drug Administration'
inds = np.linspace(0,1,dfDm.shape[0]+2,endpoint=True)[1:-1]
labels = dfDm.index.tolist()
xlim = (-0.0, 1.0)
#ylim = (-0.0, ylim[1])

ax1.axhline(y=0, lw=1, color='gray')
d_m = ax1.errorbar(inds, dfDm['mean'].values, yerr=dfDm['std'].values, color='#0000b2', marker='s', ms=8, lw=2, zorder=15)
d_f = ax1.errorbar(inds, dfDf['mean'].values, yerr=dfDf['std'].values, color='#4c4cff', marker='D', ms=8, lw=2, zorder=15)

ax1.set_title(r'Administration')
ax1.set_ylabel(r'$\nu^{u}$')
ax1.set_xticks(inds)
ax1.set_xticklabels(labels, rotation=90)

ax1.legend([d_m,d_f], [r'$g=M$',r'$g=F$'], loc=2)

ax1.set_xlim(xlim)
#ax1.set_ylim(ylim)
ax1.grid()


ax2.axhline(y=0, lw=1, color='gray')
c_m = ax2.errorbar(inds, dfCm['mean'].values, yerr=dfCm['std'].values, color='#b27300', marker='s', ms=8, lw=2, zorder=15)
c_f = ax2.errorbar(inds, dfCf['mean'].values, yerr=dfCf['std'].values, color='#ffc966', marker='D', ms=8, lw=2, zorder=15)

ax2.set_title(r'Co-administration')
ax2.set_ylabel(r'$\Psi^{u}$')
ax2.set_xticks(inds)
ax2.set_xticklabels(labels, rotation=90)

ax2.legend([c_m,c_f], [r'$g=M$',r'$g=F$'], loc=2)

ax2.set_xlim(xlim)
#ax2.set_ylim(ylim)
ax2.grid()


ax3.axhline(y=0, lw=1, color='gray')
i_m = ax3.errorbar(inds, dfIm['mean'].values, yerr=dfIm['std'].values, color='#b20000', marker='s', ms=8, lw=2, zorder=15)
i_f = ax3.errorbar(inds, dfIf['mean'].values, yerr=dfIf['std'].values, color='#ff6666', marker='D', ms=8, lw=2, zorder=15)

ax3.set_title(r'Interaction')
ax3.set_ylabel(r'$\Phi^{u}$')
ax3.set_xticks(inds)
ax3.set_xticklabels(labels, rotation=90)

ax3.legend([i_m,i_f], [r'$g=M$',r'$g=F$'], loc=2)

ax3.set_xlim(xlim)
#ax3.set_ylim(ylim)
ax3.grid()

print '--- Export Plot File ---'
#plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.97, wspace=0.05, hspace=0.05)
#plt.subplots_adjust(wspace=0.20)
plt.tight_layout()
plt.savefig('images/img-age-means.pdf', dpi=300)
plt.close()
