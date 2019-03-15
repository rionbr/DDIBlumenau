# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot CoAdmin and DDI Distribution 
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
import matplotlib.cm as cm
import itertools

import powerlaw
import numpy as np
import scipy
from scipy import interp
from scipy.stats import ttest_ind #, linregress
import statsmodels.api as sm
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import util


#
#
# Load CSVs
#
"""
dfd = pd.read_csv('results/dd_drugs.csv.gz', header=0, encoding='utf-8', nrows=None)
dfdg = dfd.groupby(['id_usuario','den']).agg({'count':'sum'}).unstack().clip(0,1)
dfdg.columns = dfdg.columns.droplevel(level=0)
drug_cols = dfdg.columns.tolist()
"""
dfu, dfc, dfi = util.dfUsersInteractionsSummary()

# Select only users with DDI
#dfu = dfu.loc[ dfu['n_ij_ddi']>0 , :]

print '>> dfu'
print dfu.head()
print '>> dfc'
print dfc.head()
print '>> dfi'
print dfi.head()

# Shuffle (men womne)
dfu = dfu.sample(frac=1)

n = dfu.shape[0]

# Remove Big Outlier
"""
threshold = 2200
print 'Outliers removed:', dfsg.loc[ (dfsg['qt_coadmin']>threshold) , : ].shape
dfsg = dfsg.loc[ (dfsg['qt_coadmin']<=threshold) , : ]
dfusg = pd.concat([dfsg, dfu], axis=1)
dfusg.reset_index(inplace=True)
"""
#print dfusg.head()
#print dfusg.shape

fig, ((axt1,axt2,axt3),(axb1,axb2,axb3))= plt.subplots(nrows=2, ncols=3, figsize=(10,6))
plt.rc('axes', titlesize=12)
plt.rc('font', size=12)
plt.rc('legend', fontsize=12)
plt.rc('legend', numpoints=1)


norm = mpl.colors.Normalize(vmin=0, vmax=80)
cmapHex = LinearSegmentedColormap.from_list('CmapHex', ['white', 'black'])
#cmapHexBlue = LinearSegmentedColormap.from_list('Blue', ['white', 'blue'])
#cmapHexRed = LinearSegmentedColormap.from_list('Red', ['white', 'red'])
cmapBlue = cm.Blues
cmapRed = cm.Reds
mM = cm.ScalarMappable(norm=norm, cmap=cmapBlue)
mF = cm.ScalarMappable(norm=norm, cmap=cmapRed)

mM.set_array(dfu['age'].values)
mF.set_array(dfu['age'].values)


def setcolor(row):
	if row['gender']=='Male':
		#return (0,0,0,1)
		return mM.to_rgba(row['age'])
	elif row['gender']=='Female':
		#return (1,1,1,1)
		return mF.to_rgba(row['age'])
colors = dfu[['gender','age']].apply(setcolor, axis=1)
#colors = np.random.rand(dfusg.shape[0])

n_i = dfu['n_i'].values
n_ij = dfu['n_ij'].values
n_ij_ddi = dfu['n_ij_ddi'].values
lw = 1
st = 120
sb = 8

sc1 = axb1.scatter(n_i+np.random.rand(n), n_ij+np.random.rand(n), s=sb, c=colors, alpha=1, marker='.', linewidths=0.4, edgecolors='None', rasterized=True)
sc2 = axb2.scatter(n_i+np.random.rand(n), n_ij_ddi+np.random.rand(n), s=sb, c=colors, alpha=1, marker='.', linewidths=0.4, edgecolors='None', rasterized=True)
sc3 = axb3.scatter(n_ij+np.random.rand(n), n_ij_ddi+np.random.rand(n), s=sb, c=colors, alpha=1, marker='.', linewidths=0.4, edgecolors='None', rasterized=True)


# INSERT 
#axb1in = inset_axes(axb1, width='25%', height='22.72%', loc=3, borderpad=0.02)
axb1in = zoomed_inset_axes(axb1, 1.25, loc=2, bbox_to_anchor=(0.001,0.999), bbox_transform=axb1.transAxes)
axb2in = zoomed_inset_axes(axb2, 1.25, loc=2, bbox_to_anchor=(0.001,0.999), bbox_transform=axb2.transAxes)
axb3in = zoomed_inset_axes(axb3, 1.25, loc=2, bbox_to_anchor=(0.001,0.999), bbox_transform=axb3.transAxes)

#
# Liner Regressions (with different exponents)
#
exp_max = 6

color = {1:'green',2:'orange',3:'grey',4:'grey',5:'grey'}

#OLS (nu,phi) Linear
r = []
print '-- Linear'
x_ = np.linspace(1, n_i.max(), 100)
for exp in np.arange(1,exp_max):
	xl = []
	xl_ = []
	for i in np.arange(1,exp+1):
		xl.append( n_i**i )
		xl_.append( x_**i )
	xl = sm.add_constant(np.column_stack( xl ))
	xl_ = sm.add_constant(np.column_stack( xl_ ))
	ols = sm.OLS(n_ij, xl).fit()
	y_ = np.dot(xl_, ols.params)
	if exp<=2:
		axb1.plot(x_, y_, color=color[exp], lw=lw, zorder=5)
		axt1.annotate('{:.3f}'.format(ols.rsquared), xy=(exp,ols.rsquared), xytext=(exp+.5,ols.rsquared-.1), fontsize=10, ha='left', va='top',
			arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=-0.15', facecolor='gray', edgecolor='gray'), zorder=10)
	r.append(('n_ij','n_i',exp, ols.rsquared))

dfr = pd.DataFrame(r, columns=['x','y','exp','r2'])
x, y = dfr['exp'].values, dfr['r2'].values
axt1.scatter(x,y, c=color.values(), s=st, marker='o', zorder=5)
axt1.plot(x,y, color='red', marker='o', lw=lw, ms=0, zorder=3)

print dfr

#OLS (nu,psi) Linear
r = []
x_ = np.linspace(1, n_i.max(), 100)
for exp in np.arange(1,exp_max):
	xl = []
	xl_ = []
	for i in np.arange(1,exp+1):
		xl.append( n_i**i )
		xl_.append( x_**i )
	xl = sm.add_constant(np.column_stack( xl ))
	xl_ = sm.add_constant(np.column_stack( xl_ ))
	ols = sm.OLS(n_ij_ddi, xl).fit()
	y_ = np.dot(xl_, ols.params)
	if exp<=2:
		axb2.plot(x_, y_, color=color[exp], lw=lw, zorder=5)
		axt2.annotate('{:.3f}'.format(ols.rsquared), xy=(exp,ols.rsquared), xytext=(exp+.5,ols.rsquared-.1), fontsize=10, ha='left', va='top',
			arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=-0.15', facecolor='gray', edgecolor='gray'), zorder=10)
	r.append(('n_ij_ddi','n_i',exp, ols.rsquared))

dfr = pd.DataFrame(r, columns=['x','y','exp','r2'])
x, y = dfr['exp'].values, dfr['r2'].values
axt2.scatter(x,y, c=color.values(), s=st, marker='o', zorder=5)
axt2.plot(x,y, color='red', marker='o', lw=lw, ms=0, zorder=3)
print dfr

#OLS (phi,psi) Linear
r = []
x_ = np.linspace(1, n_ij.max(), 100)
for exp in np.arange(1,exp_max):
	xl = []
	xl_ = []
	for i in np.arange(1,exp+1):
		xl.append( n_ij**i )
		xl_.append( x_**i )
	xl = sm.add_constant(np.column_stack( xl ))
	xl_ = sm.add_constant(np.column_stack( xl_ ))
	ols = sm.OLS(n_ij_ddi, xl).fit()
	y_ = np.dot(xl_, ols.params)
	if exp<=2:
		axb3.plot(x_, y_, color=color[exp], lw=lw, zorder=5)
		if exp<=1:
			axt3.annotate('{:.3f}'.format(ols.rsquared), xy=(exp,ols.rsquared), xytext=(exp+.5,ols.rsquared-.1), fontsize=10, ha='left', va='top',
				arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=-0.15', facecolor='gray', edgecolor='gray'), zorder=10)
	r.append(('n_ij_ddi','n_ij',exp, ols.rsquared))

dfr = pd.DataFrame(r, columns=['x','y','exp','r2'])
x, y = dfr['exp'].values, dfr['r2'].values
axt3.scatter(x,y, c=color.values(), s=st, marker='o', zorder=5)
axt3.plot(x,y, color='red', marker='o', lw=lw, ms=0, zorder=3)
print dfr

axt1.set_title(r'$\Psi^{u}$ from $\nu^{u}$')
axt2.set_title(r'$\Phi^{u}$ from $\nu^{u}$')
axt3.set_title(r'$\Phi^{u}$ from $\Psi^{u}$')

# INSERT HexBin
axb1in.tick_params(axis='both',bottom='off',left='off')
axb2in.tick_params(axis='both',bottom='off',left='off')
axb3in.tick_params(axis='both',bottom='off',left='off')

axb1in.set_xticks([])
axb1in.set_yticks([])
axb1in.set_xlim(0, 14)
axb1in.set_ylim(0, 50)
axb2in.set_xticks([])
axb2in.set_yticks([])
axb2in.set_xlim(0, 15)
axb2in.set_ylim(0, 3)
axb3in.set_xticks([])
axb3in.set_yticks([])
axb3in.set_xlim(0, 75)
axb3in.set_ylim(0, 3)

hb1 = axb1in.hexbin(n_i, n_ij, gridsize=20, bins='log', cmap=cmapHex, zorder=0)
hb2 = axb2in.hexbin(n_i, n_ij_ddi, gridsize=20, bins='log', cmap=cmapHex, zorder=0)
hb3 = axb3in.hexbin(n_ij, n_ij_ddi, gridsize=20, bins='log', cmap=cmapHex, zorder=0)

axt1.set_xticks(np.arange(1,exp_max))
axt2.set_xticks(np.arange(1,exp_max))
axt3.set_xticks(np.arange(1,exp_max))
axt1.set_xticklabels(np.arange(1,exp_max))
axt2.set_xticklabels(np.arange(1,exp_max))
axt3.set_xticklabels(np.arange(1,exp_max))


axt1.set_ylabel(r'$R^2$')
axt1.set_xlabel(r'complexity')
axt2.set_ylabel(r'$R^2$')
axt2.set_xlabel(r'complexity')
axt3.set_ylabel(r'$R^2$')
axt3.set_xlabel(r'complexity')
#
axb1.set_xlabel(r'$\nu^{u}$')
axb1.set_ylabel(r'$\Psi^{u}$')
axb2.set_xlabel(r'$\nu^{u}$')
axb2.set_ylabel(r'$\Phi^{u}$')
axb3.set_ylabel(r'$\Phi^{u}$')
axb3.set_xlabel(r'$\Psi^{u}$')

axt1.set_xlim(0,exp_max)
axt2.set_xlim(0,exp_max)
axt3.set_xlim(0,exp_max)
axt1.set_ylim(0,.9)
axt2.set_ylim(0,.9)
axt3.set_ylim(0,.9)
#
axb1.set_xlim(0,35)
axb1.set_ylim(0,300)
axb2.set_xlim(0,40)
axb2.set_ylim(0,15)
axb3.set_xlim(0,300)
axb3.set_ylim(0,15)

axt1.grid()
axt2.grid()
axt3.grid()
#
axb1.grid()
axb2.grid()
axb3.grid()

#axb1.set_aspect('equal')
#axb2.set_aspect('equal')
#axb3.set_aspect('equal')


# INSERT
mark_inset(axb1, axb1in, loc1=2, loc2=4, fc="none", ec="gray", lw=0.5)
mark_inset(axb2, axb2in, loc1=2, loc2=4, fc="none", ec="gray", lw=0.5)
mark_inset(axb3, axb3in, loc1=2, loc2=4, fc="none", ec="gray", lw=0.5)


caxb1 = make_axes_locatable(axb1).append_axes('right',size='3.5%', pad=0.05)
caxb2 = make_axes_locatable(axb2).append_axes('right',size='3.5%', pad=0.05)
caxb3 = make_axes_locatable(axb3).append_axes('right',size='3.5%', pad=0.05)

caxM = fig.add_axes([0.28, 0.04, .15, .02]) # left, bottom, width, height
caxF = fig.add_axes([0.61, 0.04, .15, .02])


cb1 = fig.colorbar(hb1, cax=caxb1)
cb2 = fig.colorbar(hb2, cax=caxb2)
cb3 = fig.colorbar(hb3, cax=caxb3)

cbM = fig.colorbar(mM, cax=caxM, orientation='horizontal', ticks=[0, 20, 40, 60, 80])
cbF = fig.colorbar(mF, cax=caxF, orientation='horizontal', ticks=[0, 20, 40, 60, 80])
cbM.ax.set_title(r'Male $y$')
cbF.ax.set_title(r'Female $y$')

plt.subplots_adjust(left=0.07, bottom=0.13, right=0.96, top=0.93, wspace=0.45, hspace=0.34)
plt.savefig('images/img-coadmin-ddi-dist.pdf', dpi=300) #, frameon=True, bbox_inches='tight', pad_inches=0.0)


