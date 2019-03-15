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
strs = ['%02d-%02d' % (x, x1-1) for x, x1 in zip(ints[:-1], ints[1:]) ] + ['80+']
def apply_age_group(x):
	g = np.digitize([x], ints, right=False)[0]
	return strs[g-1]

#
# Load Interaction / Users
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(apply_age_group=apply_age_group, cat_age_groups=strs)

dfu.reset_index(inplace=True)

# Merge Small Numbers
#dfusg['age_group'].cat.add_categories('>84', inplace=True)
#dfusg.loc[ (dfusg['age_group'].isin(['85-89','90-94','95-99','>99'])),'age_group'] = '>84'
#dfusg['age_group'].cat.remove_categories(['85-89','90-94','95-99','>99'], inplace=True)


# Male & Female
dfu_m = dfu.loc[ (dfu['gender']=='Male') , : ]
dfu_f = dfu.loc[ (dfu['gender']=='Female') , : ]


ra = []
rb = []
rc = []
for age_group, dfu_tmp in dfu.groupby('age_group'):

	ols = sm.OLS(dfu_tmp['n_ij'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	ra.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rb.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_ij'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rc.append( (age_group, r2, p, slope, bias) )

dfRa = pd.DataFrame(ra,columns=['age_group','r2','p-value','slope','bias'])
dfRb = pd.DataFrame(rb,columns=['age_group','r2','p-value','slope','bias'])
dfRc = pd.DataFrame(rc,columns=['age_group','r2','p-value','slope','bias'])

print '> dfRa'
print dfRa
print '> dfRb'
print dfRb
print '> dfRc'
print dfRc

ra_m, ra_f = [], []
rb_m, rb_f = [], []
rc_m, rc_f = [], []
for age_group, dfu_tmp in dfu_m.groupby('age_group'):

	ols = sm.OLS(dfu_tmp['n_ij'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	ra_m.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rb_m.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_ij'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rc_m.append( (age_group, r2, p, slope, bias) )

for age_group, dfu_tmp in dfu_f.groupby('age_group'):

	ols = sm.OLS(dfu_tmp['n_ij'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	ra_f.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_i'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rb_f.append( (age_group, r2, p, slope, bias) )

	ols = sm.OLS(dfu_tmp['n_ij_ddi'].values, sm.add_constant(dfu_tmp['n_ij'].values)).fit()
	r2,p = ols.rsquared, ols.pvalues[1]
	slope, bias = ols.params[1], ols.params[0]
	rc_f.append( (age_group, r2, p, slope, bias) )

dfRa_m = pd.DataFrame(ra_m, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')
dfRb_m = pd.DataFrame(rb_m, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')
dfRc_m = pd.DataFrame(rc_m, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')
#
dfRa_f = pd.DataFrame(ra_f, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')
dfRb_f = pd.DataFrame(rb_f, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')
dfRc_f = pd.DataFrame(rc_f, columns=['age_group','r2','p-value','slope','bias']).set_index('age_group')

print '> dfRc_m'
print dfRc_m
print '> dfRc_f'
print dfRc_f
#
# Plotting
#
print '- Plotting -'
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11,3))
plt.rc('font', size=12)
plt.rc('legend', fontsize=9)
plt.rc('legend', numpoints=1)



#inds = np.linspace(0, 1, dfRa.shape[0]+2, endpoint=True)[1:-1]
inds = dfRa.index.values

print 'Mean Male 20-39'
print dfRc_m.loc['20-14':'35-39','r2'].mean()
print 'Mean Female 20-39'
print dfRc_f.loc['20-14':'35-39','r2'].mean()

print 'Mean Male 20-39'
print dfRc_m.loc['55-59':'80+','r2'].mean()
print 'Mean Female 20-39'
print dfRc_f.loc['55-59':'80+','r2'].mean()

print 'Difference'
print dfRc_m['r2'] - dfRc_f['r2']

xlim = (inds[0]-.5, inds[-1]+.5)
ylim = (-0.0, 1.0)

width = 0.34
r = 0.08
lw = 1

labels = dfRa['age_group'].values

r2a_m = ax1.bar(inds-width, dfRa_m['r2'].values, width, color='#b27300', zorder=10)
r2a_f = ax1.bar(inds, dfRa_f['r2'].values, width, color='#ffc966', zorder=10)
r2a, = ax1.plot(inds, dfRa['r2'].values, color='orange', marker='o', ms=6, lw=0, zorder=20)

ax1.legend([r2a_m,r2a_f,r2a],[r'$y,g=M$',r'$y,g=F$',r'$y$'], ncol=3, handletextpad=0.4, loc=2, columnspacing=0.5, handlelength=1)

for x, row in dfRa.iterrows():
	y = row['r2']
	angle = math.atan(row['slope'])
	dx = r*math.cos(angle)
	dy = r*math.sin(angle)
	#
	#ax1.plot((x-dx,x+dx), (y-dy,y+dy), color='green', lw=lw, zorder=15)

ax1.set_title(r'$R^{2} ( \Psi^{u} , \nu^{u} ) $')
ax1.set_ylabel(r'$R^{2}$')
ax1.set_xticks(inds)
ax1.set_xticklabels(labels, rotation=90)

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.grid()


#
#
#
#a = math.radians(45)
#ax2.arrow(4, 0.4, 0.1*math.cos(a), 0.1*math.sin(a))

r2b_m = ax2.bar(inds-width, dfRb_m['r2'].values, width, color='#b20000', zorder=10)
r2b_f = ax2.bar(inds, dfRb_f['r2'].values, width, color='#ff6666', zorder=10)
r2b, = ax2.plot(inds, dfRb['r2'].values, color='red', marker='o', ms=6, lw=0, zorder=20)

ax2.legend([r2b_m,r2b_f,r2b],[r'$y,g=M$',r'$y,g=F$',r'$y$'], loc=2)

ax2.axvspan(0.5, 4.5, alpha=0.35, color='gray')
ax2.axvspan(7.5, 13.5, alpha=0.35, color='gray')


for x, row in dfRb.iterrows():	
	y = row['r2']
	angle = math.atan(row['slope'])
	dx = r*math.cos(angle)
	dy = r*math.sin(angle)
	#
	#ax2.plot((x-dx,x+dx), (y-dy,y+dy), color='green', lw=lw, zorder=15)

ax2.set_title(r'$R^{2} ( \Phi^{u} , \nu^{u} ) $')
ax2.set_ylabel(r'$R^{2}$')
ax2.set_xticks(inds)
ax2.set_xticklabels(labels, rotation=90)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.grid()


#
#
#
r2c_m = ax3.bar(inds-width, dfRc_m['r2'].values, width, color='#b20000', zorder=10)
r2c_f = ax3.bar(inds, dfRc_f['r2'].values, width, color='#ff6666', zorder=10)
r2c, = ax3.plot(inds, dfRc['r2'].values, color='red', marker='o', ms=6, lw=0, zorder=20)

ax3.legend([r2c_m,r2c_f,r2c],[r'$y,g=M$',r'$y,g=F$',r'$y$'], loc=2)

ax3.axvspan(0.5, 4.5, alpha=0.35, color='gray')
ax3.axvspan(7.5, 13.5, alpha=0.35, color='gray')

for x, row in dfRc.iterrows():
	y = row['r2']
	angle = math.atan(row['slope'])
	dx = r*math.cos(angle)
	dy = r*math.sin(angle)
	#
	#ax3.plot((x-dx,x+dx), (y-dy,y+dy), color='green', lw=lw, zorder=15)

ax3.set_title(r'$R^{2} ( \Phi^{u} , \Psi^{u} ) $')
ax3.set_ylabel(r'$R^{2}$')
ax3.set_xticks(inds)
ax3.set_xticklabels(labels, rotation=90)

ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.grid()

#axM.grid()
#axF.grid()
print '--- Export Plot File ---'
#plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.97, wspace=0.05, hspace=0.05)
#plt.subplots_adjust(wspace=0.20)
plt.tight_layout()
plt.savefig('images/img-r2-age-gender.pdf', dpi=300)
plt.close()
