# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot  results of age of users on interactin drugs
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
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.stats import norm, ttest_ind
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import util
import numpy as np


def rename_ages(x):
	if (x != '>99'):
		a,b = x.split('-')
		return ('%d to %d' % (int(a),int(b)) )
	else:
		return x

#
# Load Interaction / Users
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary()

print '>> dfu'
print dfu.head()

# Load Bnu Data
dfBnu = util.BnuData(age_per_gender=True)

# Bairro Order
sRenda = dfBnu['avg_income'].copy().sort_values(ascending=False)
print '>> sRenda'
print sRenda.head()
bairros = sRenda.index.tolist()

# Move bairro OTHER to end of list
bairros.append(bairros.pop(34))

df_M_Bnu = dfBnu.iloc[:,0:21].T
df_F_Bnu = dfBnu.iloc[:,21:42].T

# Remove Gender from Age Name
df_M_Bnu.index = df_M_Bnu.index.map(lambda x: x[2:])
df_F_Bnu.index = df_F_Bnu.index.map(lambda x: x[2:])

# Sort columns in the same order
df_M_Bnu = df_M_Bnu[bairros]
df_F_Bnu = df_F_Bnu[bairros]
df_T_Bnu = df_M_Bnu + df_F_Bnu

print '>> df_T_Bnu'
print df_T_Bnu.iloc[:5,:5]
#
# Test Dispensation 
#
dfG_T = dfu.reset_index().groupby(['age_group','hood'], sort=False).agg({'id_user':'count'}).rename(columns={'id_user':'count'})
print dfG_T.head()
dfG_T = dfG_T.unstack(level=1)
dfG_T.columns = dfG_T.columns.droplevel(0)

dfR_T = dfG_T / df_T_Bnu
print '>> dfR_T'
print dfR_T.iloc[:5,:5]

#
mask = dfR_T.columns.isin(['JARDIM BLUMENAU','BOM RETIRO','VICTOR KONDER','VILA FORMOSA'])
a = dfR_T.loc[ : , mask ].stack().clip(0,1).values
b = dfR_T.loc[ : , ~mask ].stack().clip(0,1).values
tstat,pvalue = ttest_ind( a , b, equal_var=True)
print 'T-Test richest neighborhoods have fewer patients'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)
#
# Separate Groups
# 
df_M = dfu.loc[ (dfu['gender']=='Male')   , ['age_group','hood'] ]
df_F = dfu.loc[ (dfu['gender']=='Female') , ['age_group','hood'] ]
#

dfG_M = df_M.reset_index().groupby(['age_group','hood'], sort=False).agg({'id_user':'count'}).rename(columns={'id_user':'count_m'})
dfG_M = dfG_M.unstack(level=1)
dfG_M.columns = dfG_M.columns.droplevel(0)
#
dfG_F = df_F.reset_index().groupby(['age_group','hood'], sort=False).agg({'id_user':'count'}).rename(columns={'id_user':'count_f'})
dfG_F = dfG_F.unstack(level=1)
dfG_F.columns = dfG_F.columns.droplevel(0)

# Sort columns in the same order
dfG_M = dfG_M[bairros]
dfG_F = dfG_F[bairros]

# Divide by Population
dfR_M = dfG_M / df_M_Bnu
dfR_F = dfG_F / df_F_Bnu



a = dfR_M.loc[ : , mask ].stack().clip(0,1).values
b = dfR_M.loc[ : , ~mask ].stack().clip(0,1).values
tstat,pvalue = ttest_ind( a , b, equal_var=True)
print 'T-Test MALES in richest neighborhoods have fewer patients'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)
a = dfR_F.loc[ : , mask ].stack().clip(0,1).values
b = dfR_F.loc[ : , ~mask ].stack().clip(0,1).values
tstat,pvalue = ttest_ind( a , b, equal_var=True)
print 'T-Test FEMALES in richest neighborhoods have fewer patients'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)

#
# Test Females 45-75 from Victor Koner against Jardim Blumenau
#
print '---'
a = dfR_F.loc['45-49':'70-74' , ['JARDIM BLUMENAU'] ].stack().clip(0,1).values
b = dfR_F.loc['45-49':'70-74' , ['VICTOR KONDER','BOM RETIRO'] ].stack().clip(0,1).values
c = dfR_F.loc['45-49':'70-74' , ~mask ].stack().clip(0,1).values
tstat,pvalue = ttest_ind( a, c, equal_var=True)
print 'T-Test (45-74 in JARDIM BLUMENAU - ~MASK)'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)
tstat,pvalue = ttest_ind( b, c, equal_var=True)
print 'T-Test (45-74 in VICTORK & BR - ~MASK)'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)
tstat,pvalue = ttest_ind( a, b, equal_var=True)
print 'T-Test (45-74 in JARDIM BLUMENAU - VK-BR)'
print 'tstat: %.2f | p-value: %.e' % (tstat, pvalue)



dfR_M = dfR_M.fillna(-1)
dfR_F = dfR_F.fillna(-1)
print dfR_M.iloc[:5,:5]
print dfR_M[dfR_M.isnull().any(axis=1)]


#
# Plotting
#
print '- Plotting -'
nrows,ncols = 100, 100
dx, dy = 1, 1
#figsize = plt.figaspect(float(dy * nrows) / float(dx * ncols))
figsize = (9,12)
fig = plt.figure(figsize=figsize)
gs = gridspec.GridSpec(nrows,ncols)

axR = plt.subplot(gs[0:16  , 0:95])
axM  = plt.subplot(gs[20:58, 0:95])
axMc = plt.subplot(gs[20:58, 97:100])
axF  = plt.subplot(gs[62:100, 0:95])
axFc = plt.subplot(gs[62:100, 97:100])

#axMc = fig.add_axes([0.5,0.3,0.01,0.6])
axMc.axis('off')
axFc.axis('off')
#fig.add_axes(axM,axF,axMt,axFt,axFc)

#fig.suptitle('')
plt.rc('font', size=12)
plt.rc('legend', fontsize=9)
plt.rc('legend', numpoints=1)
#
# 
#
m,n = dfG_M.shape
xmin, ymin = 1,1
xmax, ymax = n+1, m+1
dx, dy = 1,1
X2, Y2 = np.meshgrid(np.arange(xmin,xmax+dx,dx)-dx/2.,np.arange(ymin,ymax+dy,dy)-dy/2.)

cmM = LinearSegmentedColormap.from_list('cmM', [(1.00, 1.00, 1.00),(1.00, 0.50, 0.00),(0.00,0.00,1.00),(0.00,0.00,0.54),(0.00, 0.00, 0.00)], N=256)
#cmM = plt.get_cmap('Blues')
#cmM.set_bad(color='cyan', alpha=1.)
cmM.set_under(color='cyan', alpha=1.)
cmM.set_over(color='green', alpha=1.)
hM = axM.pcolormesh(X2, Y2, dfR_M.values, vmin=0, vmax=1, cmap=cmM, edgecolors='None')
bH = fig.colorbar(hM, ax=axMc, fraction=1, extend='both')

cmF = LinearSegmentedColormap.from_list('cmF', [(1.00, 1.00, 1.00),(1.00, 0.50, 0.00),(1.00,0.00,0.00),(0.00,0.00,0.54),(0.00, 0.00, 0.00)], N=256)
#cmF = plt.get_cmap('Reds')
#cmF.set_bad(color='green', alpha=1.)
cmF.set_under(color='cyan', alpha=1.)
cmF.set_over(color='green', alpha=1.)

hF = axF.pcolormesh(X2, Y2, dfR_F.values, vmin=0, vmax=1, cmap=cmF, edgecolors='None')
bF = plt.colorbar(hF, ax=axFc, fraction=1, extend='both')

axR.set_title(r'Neighborhood average income (R$/month)')
axM.set_title(r'Males patients $U^{N,y,g=M} / \Omega^{N,y,g=M}$')
axF.set_title(r'Females patients $U^{N,y,g=F}/ \Omega^{N,y,g=F}$')

axM.set_xlim(X2.min(), X2.max())
axM.set_ylim(Y2.min(), Y2.max())
axF.set_xlim(X2.min(), X2.max())
axF.set_ylim(Y2.min(), Y2.max())

age_labels = map(lambda x : rename_ages(x), dfR_M.index.values)

axM.set_xticks(np.arange(xmin,xmax,dx))
axM.set_yticks(np.arange(ymin,ymax,dy))
axM.set_xticklabels([], rotation=90)
axM.set_yticklabels(age_labels)

axF.set_xticks(np.arange(xmin,xmax,dx))
axF.set_yticks(np.arange(ymin,ymax,dy))
axF.set_xticklabels(dfR_F.columns.values, rotation=90)
axF.set_yticklabels(age_labels)

"""axR.set_aspect('data')

axMc.set_aspect('equal')
axF.set_aspect('equal')
axFc.set_aspect('equal')
"""

#axF.tick_params(axis='y', which='major', pad=32)
#axM.tick_params(axis='both', which='major', pad=28)
#for tick in axF.yaxis.get_major_ticks():
#	tick.label1.set_horizontalalignment('center')

#axM.invert_xaxis()
# Change Legend labels

#
#
#
axR.plot(range(len(bairros)), sRenda.values, c='orange', lw=2, marker='o', ms=10)
#
axR.set_xlim(X2.min()-1, X2.max()-1)
axR.set_ylim(0, 4000)
axR.set_yticks( np.linspace(sRenda.min(), sRenda.max(), 5) )
axR.set_yticklabels( np.linspace(sRenda.min(), sRenda.max(), 5) )
axR.set_xticks(np.arange(xmin-1,xmax-1,dx))
axR.set_xticklabels([])
axR.grid()
axR.set_ylabel(r'$R\$$')


#axM.grid()
#axF.grid()
print '--- Export Plot File ---'
plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.97, wspace=0.05, hspace=0.05)
#plt.subplots_adjust(wspace=0.20)
#plt.tight_layout()
plt.savefig('images/img-age-hood-income.pdf', dpi=300)
plt.close()
