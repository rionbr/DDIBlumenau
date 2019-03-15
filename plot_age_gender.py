# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of age of users on interactin drugs
#
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
import string
import numpy as np
from scipy.stats import norm, ttest_ind, ks_2samp
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)
pd.set_option('display.precision', 4)
import util
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def JSD(P, Q):
	_P = P / norm(P, ord=1)
	_Q = Q / norm(Q, ord=1)
	_M = 0.5 * (_P + _Q)
	return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

#
# Load CSVs
#
dfR = pd.read_csv('csv/age.csv', index_col=0, encoding='utf-8')
dfR_m = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfR_f = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')

df00_89_m = dfR_m.iloc[ 0:18, : ]
df90_pl_m = dfR_m.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfRs_m = pd.concat([df00_89_m, df90_pl_m], axis=0)


df00_89_f = dfR_f.iloc[ 0:18, : ]
df90_pl_f = dfR_f.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfRs_f = pd.concat([df00_89_f, df90_pl_f], axis=0)



print '>> dfRs_f'
print dfRs_f

# Load Bnu Data
dfCenso = util.dfCenso(age_per_gender=False)
df_Bnu = dfCenso.iloc[:,7:49].sum(axis=0).to_frame(name='p^{y}')
df_Bnu['P(p^{y})'] = df_Bnu['p^{y}'] / df_Bnu['p^{y}'].sum()

dfCenso = util.dfCenso(age_per_gender=True)
# All age ranges
dfBnu_m = dfCenso.iloc[:,7:28].sum(axis=0).to_frame(name='p')
dfBnu_f = dfCenso.iloc[:,28:49].sum(axis=0).to_frame(name='p')
# A short age range
dfBnu00_89_m = dfBnu_m.iloc[ 0:18, : ]
dfBnu90_pl_m = dfBnu_m.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfBnus_m = pd.concat([dfBnu00_89_m, dfBnu90_pl_m], axis=0)

dfBnu00_89_f = dfBnu_f.iloc[ 0:18, : ]
dfBnu90_pl_f = dfBnu_f.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfBnus_f = pd.concat([dfBnu00_89_f, dfBnu90_pl_f], axis=0)


dfBnus_m['P(p)'] = dfBnus_m['p'] / (dfBnus_m['p'].sum() + dfBnus_f['p'].sum())
dfBnus_f['P(p)'] = dfBnus_f['p'] / (dfBnus_m['p'].sum() + dfBnus_f['p'].sum())

print '>> dfBnus_f'
print dfBnus_f
#
# Plotting
#
print '- Plotting -'
fig, (axM, axF) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(8,5))
#axF = plt.subplot2grid( (1,2), (0,1))


#fig.suptitle('')
plt.rc('font', size=12)
plt.rc('legend', fontsize=12)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)



print dfR.sum(axis=0)
xmin, xmax = dfR.min().min(), dfR.max().max()
xmax = max(abs(xmin), abs(xmax))+0.01

ind = np.arange(len(dfRs_m.index))
width = 0.30

indfill = list(ind)
indfill[0] = -0.5
indfill[-1] = 21

ms = 8

#
# Kolmogorov–Smirnov test male and female interaction differ
#
#print 'K-S Test '
#print ks_2samp(dfR['i^{a,g=m}'].values, dfR['i^{a,g=f}'].values)


# Fill Between (Pop.)
p_m = axM.fill_betweenx(indfill, dfBnus_m['P(p)'].values, color='#b2d8b2', zorder=2, lw=2, edgecolor='#99cc99')
p_f = axF.fill_betweenx(indfill, dfBnus_f['P(p)'].values, color='#b2d8b2', zorder=2, lw=2, edgecolor='#99cc99')

# Normalizaed Plots
d_m, = axM.plot(dfRs_m['P(u)'].values, ind, color='#0000b2', lw=2, marker='s', ms=ms, zorder=5)
d_f, = axF.plot(dfRs_f['P(u)'].values, ind, color='#4c4cff', lw=2, marker='D', ms=ms, zorder=5)

c_m, = axM.plot(dfRs_m['P(c)'].values, ind, color='#b27300', lw=2, marker='s', ms=ms, linestyle='dashed', zorder=6)
c_f, = axF.plot(dfRs_f['P(c)'].values, ind, color='#ffc966', lw=2, marker='D', ms=ms, linestyle='dashed', zorder=6)

i_m, = axM.plot(dfRs_m['P(i)'].values, ind, color='#b20000', lw=2, marker='s', ms=ms, linestyle='dashed', zorder=7)
i_f, = axF.plot(dfRs_f['P(i)'].values, ind, color='#ff6666', lw=2, marker='D', ms=ms, linestyle='dashed', zorder=7)

# Conditional Probability
#axM.plot(dfR['P_m(I|D)'].values, ind, color='#ff8000', lw=2, marker='o', ms=6)
#axF.plot(dfR['P_f(I|D)'].values, ind, color='#ff8000', lw=2, marker='o', ms=6, label='P(I|D)')

axM.set_title(r'Men')
axF.set_title(r'Women')

axM.set_ylim(-0.5, len(ind)-.5)
axF.set_ylim(-0.5, len(ind)-.5)

#axM.set_xlim(0, xmax)
#axF.set_xlim(0, xmax)

axM.set_yticks(ind)
axF.set_yticks(ind)
axM.set_yticklabels([])
axF.set_yticklabels(dfRs_f.index)

axF.tick_params(axis='y', which='major', pad=32)
#axM.tick_params(axis='both', which='major', pad=28)
for tick in axF.yaxis.get_major_ticks():
	tick.label1.set_horizontalalignment('center')

axM.invert_xaxis()
# Change Legend labels


m_handles, m_labels = axF.get_legend_handles_labels()
f_handles, f_labels = axF.get_legend_handles_labels()
#handles = [(m,f) for m,f in zip(m_handles,f_handles)]
#labels = [(m,f) for m,f in zip(m_labels,f_labels)]

handles = [(d_m,d_f),(c_m,c_f),(i_m,i_f),(p_m,p_f)]
labels = [r'$P(U^{\nu>0,y,g})$', r'$P(U^{\Psi,y,g})$', r'$P(U^{\Phi,y,g})$', r'$P(\Omega^{y,g})$']
axF.legend(handles, labels, loc=1)


#print 'Correlation'
#print np.round( dfR['age_i'].corr(dfR['age_c']) , 4)
axM.grid()
axF.grid()
print '--- Export Plot File ---'
plt.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.94, wspace=0.26, hspace=0.0)
#plt.subplots_adjust(wspace=0.20)
#plt.tight_layout()
plt.savefig('images/img-age-gender.pdf', dpi=300)
plt.close()
