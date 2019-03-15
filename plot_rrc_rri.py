# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
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
dfR_m = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfR_f = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')
dfR_n = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)
#
#dfRs = pd.read_csv('csv/age_short.csv', index_col=0, encoding='utf-8')
#dfRs_m = pd.read_csv('csv/age_male_short.csv', index_col=0, encoding='utf-8')
#dfRs_f = pd.read_csv('csv/age_female_short.csv', index_col=0, encoding='utf-8')


# Sum Males + Females
#dfR_n['u^{ind}'] = dfR_n['u^{F,ind}'] + dfR_n['u^{M,ind}']
#dfR_n['u^{const}'] = dfR_n['u^{F,const}'] + dfR_n['u^{M,const}']
dfR_n['u^{rnd}'] = dfR_n['u^{i,F}_{rnd}'] + dfR_n['u^{i,M}_{rnd}']

n_user = dfR['u'].sum() # This is only valid if the sample of users is 100%
n_runs = dfR_n['run'].max()

# Group all the Runs in the NullModel
dfR_n = dfR_n.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	#('u^{ind}',['mean','std']),
	#('u^{F,ind}',['mean','std']),
	#('u^{M,ind}',['mean','std']),
	#('u^{const}',['mean','std']),
	#('u^{F,const}',['mean','std']),
	#('u^{M,const}',['mean','std']),
	('u^{rnd}',['mean','std']),
	('u^{i,F}_{rnd}',['mean','std']),
	('u^{i,M}_{rnd}',['mean','std'])
	]))


print '>> dfR'
print dfR
print '>> dfRs_m'
print dfR_m
print '>> dfRs_f'
print dfR_f
print '>> dfR_null'
print dfR_n #.head()
print '>> dfRs_m'
print dfR_m.head()
print '>> dfRs_f'
print dfR_f.head()

dfR_n.columns = ['-'.join(col).strip() for col in dfR_n.columns.values]

# Complete
dfR['P(u)'] = dfR['u'] / n_user
dfR['P(c)'] = dfR['u^{c}'] / n_user
dfR['P(i)'] = dfR['u^{i}'] / n_user


# Short
#dfRs['P(u)'] = dfRs['u'] / dfRs['u'].sum()
#dfRs['P(c)'] = dfRs['u^{c}'] / dfRs['u'].sum()
#dfRs['P(i)'] = dfRs['u^{i}'] / dfRs['u'].sum()
#dfRs['RRC^{y}'] = (dfRs['P(c)'] / dfRs['P(u)']) / (dfRs.loc['00-19','P(c)']/dfRs.loc['00-19','P(u)'])
#dfRs['RRI^{y}'] = (dfRs['P(i)'] / dfRs['P(u)']) / (dfRs.loc['00-19','P(i)']/dfRs.loc['00-19','P(u)'])

# Because RRC^{m} and RRC^{f} are computed independently, to make them relatable we need this

# Short MAle
#dfRs_m['P(u)'] = dfRs_m['u'] / dfRs_m['u'].sum()
#dfRs_m['P(c)'] = dfRs_m['u^{c}'] / dfRs_m['u'].sum()
#dfRs_m['P(i)'] = dfRs_m['u^{i}'] / dfRs_m['u'].sum()
dfR_m['RRC^{y}'] = (dfR_m['P(c)'] / dfR_m['P(u)']) / (dfR_m.loc['00-04':'15-19','P(c)'].sum() / dfR_m.loc['00-04':'15-19','P(u)'].sum())
dfR_m['RRI^{y}'] = (dfR_m['P(i)'] / dfR_m['P(u)']) / (dfR_m.loc['00-04':'15-19','P(i)'].sum() / dfR_m.loc['00-04':'15-19','P(u)'].sum() )

# Short Female
#dfRs_f['P(u)'] = dfRs_f['u'] / dfRs_f['u'].sum()
#dfRs_f['P(c)'] = dfRs_f['u^{c}'] / dfRs_f['u'].sum()
#dfRs_f['P(i)'] = dfRs_f['u^{i}'] / dfRs_f['u'].sum()
dfR_f['RRC^{y}'] = (dfR_f['P(c)'] / dfR_f['P(u)']) / (dfR_m.loc['00-04':'15-19','P(c)'].sum() / dfR_m.loc['00-04':'15-19','P(u)'].sum() )
dfR_f['RRI^{y}'] = (dfR_f['P(i)'] / dfR_f['P(u)']) / (dfR_m.loc['00-04':'15-19','P(i)'].sum() / dfR_m.loc['00-04':'15-19','P(u)'].sum() )

#
# Null Models Computation - Short
#
# Compute the dfRs_n
print dfR_n
dfR_n00_19 = dfR_n.iloc[ 0:4 , : ].sum(axis=0).to_frame(name='00-19').T
dfR_n20_79 = dfR_n.iloc[ 4:16, : ]
dfR_n80_pl = dfR_n.iloc[ 16: , : ].sum(axis=0).to_frame(name='80+').T
dfRs_n = pd.concat([dfR_n00_19, dfR_n20_79, dfR_n80_pl], axis=0)

# Adding number of Male & Femals - Only true if random sample is 100%
dfRs_n['u^{i,M}'] = dfR_m['u']
dfRs_n['u^{i,F}'] = dfR_f['u']
print dfRs_n

#
dfRs_n['P(u)'] = dfRs_n['u-mean'] / n_user
#dfRs_n['P(i)_ind'] = dfRs_n['u^{ind}'] / n_user
#dfRs_n['P(i)_const'] = dfRs_n['u^{const}'] / n_user
dfRs_n['P(i)_rnd'] = dfRs_n['u^{rnd}-mean'] / n_user

# Gender
dfRs_n['P(F)'] = dfRs_n['u^{F}'] / n_user
dfRs_n['P(M)'] = dfRs_n['u^{M}'] / n_user
dfRs_n['P(i,F)_rnd'] = dfRs_n['u^{F,rnd}-mean'] / n_user
dfRs_n['P(i,M)_rnd'] = dfRs_n['u^{M,rnd}-mean'] / n_user
dfRs_n['RRI^{y,F}_rnd'] = (dfRs_n['P(i,F)_rnd'] / dfRs_n['P(F)']) / (dfRs_f.loc['00-19','P(i)'] / dfRs_n.loc['00-19','P(F)'])
dfRs_n['RRI^{y,M}_rnd'] = (dfRs_n['P(i,M)_rnd'] / dfRs_n['P(M)']) / (dfRs_m.loc['00-19','P(i)'] / dfRs_n.loc['00-19','P(M)'])

# Calculate RRI from H_0 models
#dfRs_n['RRI^{y}_ind'] = (dfRs_n['P(i)_ind'] / dfRs_n['P(u)']) / (dfRs_n.loc['00-19','P(i)_ind']/dfRs_n.loc['00-19','P(u)'])
#dfRs_n['RRI^{y}_const'] = (dfRs_n['P(i)_const'] / dfRs_n['P(u)']) / (dfRs_n.loc['00-19','P(i)_const']/dfRs_n.loc['00-19','P(u)'])
dfRs_n['RRI^{y}_rnd'] = (dfRs_n['P(i)_rnd'] / dfRs_n['P(u)']) / (dfRs_n.loc['00-19','P(i)_rnd'] / dfRs_n.loc['00-19','P(u)'])

# Calculare RRI from real data
#dfRs_n['RRI^{y}_ind'] = (dfRs_n['P(i)_ind'] / dfRs_n['P(u)']) / (dfRs.loc['00-19','P(i)']/dfRs.loc['00-19','P(u)'])
#dfRs_n['RRI^{y}_const'] = (dfRs_n['P(i)_const'] / dfRs_n['P(u)']) / (dfRs.loc['00-19','P(i)']/dfRs.loc['00-19','P(u)'])
dfRs_n['RRI^{y}_rnd_real'] = (dfRs_n['P(i)_rnd'] / dfRs_n['P(u)']) / (dfRs.loc['00-19','P(i)'] / dfRs.loc['00-19','P(u)']) # Divide by actual data

# Confidence Interval
dfRs_n[['u^{rnd}-ci_min','u^{rnd}-ci_max']] = dfRs_n[['u^{rnd}-mean','u^{rnd}-std']].apply(calc_conf_interval, axis=1)
dfRs_n['P(i)_rnd-ci_min'] = dfRs_n['u^{rnd}-ci_min'] / dfRs_n['u-mean'].sum()
dfRs_n['P(i)_rnd-ci_max'] = dfRs_n['u^{rnd}-ci_max'] / dfRs_n['u-mean'].sum()
dfRs_n['RRI^{y}_rnd_real-ci_min'] = (dfRs_n['P(i)_rnd-ci_min'] / dfRs_n['P(u)']) / ( dfRs.loc['00-19','P(i)'].sum() / dfRs.loc['00-19','P(u)'].sum() )
dfRs_n['RRI^{y}_rnd_real-ci_max'] = (dfRs_n['P(i)_rnd-ci_max'] / dfRs_n['P(u)']) / ( dfRs.loc['00-19','P(i)'].sum() / dfRs.loc['00-19','P(u)'].sum() )



print '>> dfRs_m'
print dfRs_m
print '>> dfRs_f'
print dfRs_f
print '>> dfRs_n'
print dfRs_n

print '>> dfRs_n[RRI^{y}_rnd_real]'
print dfRs_n['RRI^{y}_rnd_real']
#
# Plot Sequence of Top Interactions
#
print '- Plotting -'
fig = plt.figure(figsize=(16,11))
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)

signi = []
"""
print '-ChiSquare for: RRI x RRI_ind'
chisq, pvalue = chisquare(dfRs_n['u^{ind}'].values, f_exp=dfRs['u^{i}'].values)
signi.append( ('RRI_ind',chisq,pvalue))
print 'Chi-Sqr: %.4f | p-value: %s\n' % (chisq, pvalue)

print '-ChiSquare for: RRI x RRI_const'
chisq, pvalue = chisquare(dfRs_n['u^{const}'].values, f_exp=dfRs['u^{i}'].values)
signi.append( ('RRI_const',chisq,pvalue))
print 'Chi-Sqr: %.4f | p-value: %s\n' % (chisq, pvalue)
"""
print '-ChiSquare for: RRI x RRI_rnd'
chisq, pvalue = chisquare(dfRs_n['u^{rnd}-mean'].values, f_exp=dfRs['u^{i}'].values)
signi.append( ('RRI_rnd',chisq,pvalue))
print 'Chi-Sqr: %.4f | p-value: %s\n' % (chisq, pvalue)

"""
print '-K-S for: RRI x RRI_ind'
stat, pvalue = ks_2samp(dfRs['u^{i}'].values, dfRs_n['u^{ind}'].values)
signi.append( ('RRI_ind',stat,pvalue))
print 'K-S: %.4f | p-value: %.4f\n' % (stat,pvalue)

print '-K-S for: RRI x RRI_const'
stat, pvalue = ks_2samp(dfRs['u^{i}'].values, dfRs_n['u^{const}'].values)
signi.append( ('RRI_const',stat,pvalue))
print 'K-S: %.4f | p-value: %.4f\n' % (stat,pvalue)

print '-K-S for: RRI x RRI_rnd'
stat, pvalue = ks_2samp(dfRs['u^{i}'].values, dfRs_n['u^{rnd}'].values)
signi.append( ('RRI_rnd',stat,pvalue))
print 'K-S: %.4f | p-value: %.4f\n' % (stat,pvalue)
"""

print pd.DataFrame(signi, columns=['model','stat','p-value']).to_latex()



print '--- Plotting ---'
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11,3))

width = 0.33
ms = 8
ms_t = 5
ms_b = 10
lw=1.2
ageinds = np.arange(0, dfRs.shape[0])
agelabels = dfRs.index.values

rrc, = ax1.plot(ageinds, dfRs['RRC^{y}'].values, color='orange', marker='o', lw=0, markersize=ms, zorder=5)
rri, = ax1.plot(ageinds, dfRs['RRI^{y}'].values, color='red', marker='o', lw=0, markersize=ms, zorder=5)

#rri_ind, = ax1.plot(ageinds, dfRs_n['RRI^{y}_ind'].values, color='tomato', marker='*', lw=0, markersize=(ms/2)+2, zorder=8)
#rri_const, = ax1.plot(ageinds, dfRs_n['RRI^{y}_const'].values, color='indianred', marker='^', lw=0, markersize=ms/2, zorder=7)
rri_rnd, = ax1.plot(ageinds, dfRs_n['RRI^{y}_rnd_real'].values, color='red', marker='*', lw=0, markersize=ms, zorder=6)
ax1.fill_between(ageinds, y1=dfRs_n['RRI^{y}_rnd_real-ci_min'].values, y2=dfRs_n['RRI^{y}_rnd_real-ci_max'].values, color='lightpink', edgecolor='lightgray', lw=.5)


#Curve Fit
print '> Curve Fitting'
def linear(x,b0,bias):
	return (b0*x)+bias
def quadratic(x,b0,b1,bias):
	return b0*(x**2)+(b1*x)+bias
def cubic(x,b0,b1,b2,bias):
	return b0*(x**3)+b1*(x**2)+(b2*x)+bias


#
# RRC
#
y_rrc = dfRs['RRC^{y}'].values
x = np.arange(len(y_rrc))
# x_ is used to plot the curve after
x_ = np.linspace(x[0],x[-1],len(x)*10)


# RRC Linear Model
print '> RRC Linear Model'
Xc = sm.add_constant( x )
rrc_l_model = sm.OLS(y_rrc, Xc)
rrc_l_model_result = rrc_l_model.fit()
print rrc_l_model_result.summary()


# RRC Cubic Model
print '> RRC CUBIC'
Xc = sm.add_constant( np.column_stack( [x**3, x**2, x] ) )
rrc_c_model = sm.OLS(y_rrc, Xc)
rrc_c_model_result = rrc_c_model.fit()
print rrc_c_model_result.summary()
Xc_ = sm.add_constant( np.column_stack( [x_**3, x_**2, x_] ) )
y_rrc_ = np.dot(Xc_, rrc_c_model_result.params)

rrc_f = ax1.plot(x_, y_rrc_, color='orange', lw=2, zorder=3)
rrc_f = ax2.plot(x_, y_rrc_, color='orange', lw=2, zorder=3)


anova_result = anova_lm(rrc_l_model_result, rrc_c_model_result)
print anova_result
#
# RRI
#
print '- RRI'
y_rri = dfRs['RRI^{y}'].values
x = np.arange(len(y_rri))

# RRI Cubic Model
print '> RRI CUBIC '
Xi = sm.add_constant( np.column_stack( [x**3, x**2, x] ) )
rri_model = sm.OLS(y_rri, Xi)
rri_model_result = rri_model.fit()
print rri_model_result.summary()
Xi_ = np.column_stack( [x_**3, x_**2, x_] )
Xi_ = sm.add_constant(Xi_)
y_rri_ = np.dot(Xi_, rri_model_result.params)


rri_f = ax1.plot(x_, y_rri_, color='red', lw=2, zorder=3)



#ax1.legend([rrc,rri,rri_ind,rri_const,rri_rnd],[r'$RRC^{y}$',r'$RRI^{y}$',r'$H^{ind}_{0}$',r'$H^{const}_{0}$',r'$H^{rnd}_{0}$'], loc=2, handletextpad=0.0)
ax1.legend([rrc,rri,rri_rnd],[r'$RRC^{y}$',r'$RRI^{y}$',r'$H^{rnd}_{0}$'], loc=2, handletextpad=0.0)

rrc_m, = ax2.plot(ageinds, dfRs_m['RRC^{y}'].values, color='#b27300', marker='s', markersize=ms, linestyle='dashed', zorder=5) # dark orange
rrc_f, = ax2.plot(ageinds, dfRs_f['RRC^{y}'].values, color='#ffc966', marker='D', markersize=ms, linestyle='dashed', zorder=5) # light orange

ax2.axvspan(0.5, 5.5, alpha=0.35, color='gray')
ax2.legend([rrc_f, rrc_m], [r'$RRC^{y,F}$',r'$RRC^{y,M}$'], loc=2, handletextpad=0.0)


ym, yf = dfRs_m.loc['60-64','RRI^{y}'].item(), dfRs_f.loc['60-64','RRI^{y}'].item()
diff = yf - ym
ymid = (yf + ym) / 2
ax3.plot((9,9), (yf, ym), color='#00b200', lw=2)
ax3.annotate('%.0fx' % diff, xy=(9, ymid), xytext=(6,72), fontsize=10,
	arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=-0.15', facecolor='gray', edgecolor='gray'))

rri_m, = ax3.plot(ageinds, dfRs_m['RRI^{y}'].values, color='#b20000', marker='s', markersize=ms, linestyle='dashed', zorder=5) # dark red
rri_f, = ax3.plot(ageinds, dfRs_f['RRI^{y}'].values, color='#ff6666', marker='D', markersize=ms, linestyle='dashed', zorder=5) #light red

rri_m_rnd, = ax3.plot(ageinds, dfRs_n['RRI^{y,F}_rnd'].values, color='#b20000', marker='P', markersize=(ms/2)+2, linestyle='', zorder=6) # dark red
rri_f_rnd, = ax3.plot(ageinds, dfRs_n['RRI^{y,M}_rnd'].values, color='#ff6666', marker='X', markersize=(ms/2)+2, linestyle='', zorder=6) #light red


ax3.axvspan(6.5, 9.5, alpha=0.35, color='gray')
ax3.legend([rri_f, rri_m, rri_f_rnd, rri_m_rnd], [r'$RRI^{y,F}$',r'$RRI^{y,M}$',r'$H^{rnd,F}_0$',r'$H^{rnd,M}_0$'], loc=2, handletextpad=0.0)

ax1.set_title(r'$RRC^{y}$ and $RRI^{y}$')
ax1.set_xticks(ageinds)
ax1.set_xticklabels(agelabels, rotation=90)

ax2.set_title(r'$RRC^{y,g}$')
ax2.set_xticks(ageinds)
ax2.set_xticklabels(agelabels, rotation=90)

ax3.set_title(r'$RRI^{y,g}$')
ax3.set_xticks(ageinds)
ax3.set_xticklabels(agelabels, rotation=90)

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlim(-1,len(ageinds))
#ax1.set_ylim(0,30)
ax2.set_xlim(-1,len(ageinds))
ax3.set_xlim(-1,len(ageinds))

print 'Export Plot File'
#plt.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.92, wspace=0.20, hspace=0.20)
plt.tight_layout()
plt.savefig('images/img-rrc-rri.pdf', dpi=300)
plt.close()



