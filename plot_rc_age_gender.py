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
#dfR = pd.read_csv('csv/age_short.csv', index_col=0, encoding='utf-8')

n_user = dfR['u'].sum()
n_user_female = dfF['u'].sum()
n_user_male = dfM['u'].sum()
print 'Number of users: {:d}'.format(n_user)

dfR00_89 = dfR.iloc[ 0:18, 0:4 ]
dfR90_pl = dfR.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfR = pd.concat([dfR00_89, dfR90_pl], axis=0)

dfF00_89 = dfF.iloc[ 0:18, 0:4 ]
dfF90_pl = dfF.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfF = pd.concat([dfF00_89, dfF90_pl], axis=0)

dfM00_89 = dfM.iloc[ 0:18, 0:4 ]
dfM90_pl = dfM.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfM = pd.concat([dfM00_89, dfM90_pl], axis=0)

dfR['RC^{y}'] = dfR['u^{c}'] / dfR['u^{n2}']
dfF['RC^{y}'] = dfF['u^{c}'] / dfF['u^{n2}']
dfM['RC^{y}'] = dfM['u^{c}'] / dfM['u^{n2}']

print '>> dfR'
print dfR
print '>> dfM'
print dfM
print '>> dfF'
print dfF

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

rc_f, = ax.plot(ageinds, dfF['RC^{y}'].values, color='#ffc966', marker='D', markersize=ms, lw=lw, ls=ls, zorder=5)
rc_m, = ax.plot(ageinds, dfM['RC^{y}'].values, color='#b27300', marker='s', markersize=ms, lw=lw, ls=ls, zorder=5)

ax.axvspan(2.5, 6.5, alpha=0.35, color='gray')

#
#Curve Fitting
#
print '--- Curve Fitting ---'

#
# RRC
#
print '> RC'
y_rc = dfR['RC^{y}'].values
x = np.arange(len(y_rc))
# x_ is used to plot the curve after
x_ = np.linspace(x[0],x[-1],len(x)*10)


# RRC Linear Model
print '> RC Linear Model'
Xc = sm.add_constant( x )
rc_l_model = sm.OLS(y_rc, Xc)
rc_l_model_result = rc_l_model.fit()
#print rc_l_model_result.summary()


# RRC Cubic Model
print '> RC CUBIC'
Xc = sm.add_constant( np.column_stack( [x**3, x**2, x] ) )
rc_c_model = sm.OLS(y_rc, Xc)
rc_c_model_result = rc_c_model.fit()
print rc_c_model_result.summary()
Xc_ = sm.add_constant( np.column_stack( [x_**3, x_**2, x_] ) )
y_rc_ = np.dot(Xc_, rc_c_model_result.params)
#rc_f_cubic = ax.plot(x_, y_rc_, color='orange', lw=2, zorder=3)

# ANOVA
anova_result = anova_lm(rc_l_model_result, rc_c_model_result)
print anova_result



ax.legend([rc_f,rc_m],[r'$RC^{[y_1,y_2],F}$',r'$RC^{[y_1,y_2],M}$'],
	loc='upper left', handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1)

ax.set_title(r'$RC^{[y_1,y_2],g}$')
#ax.text(0, 1.09, r'$\frac{ P(\Psi^{u}>0|u \in U^{[y_1,y_2],g}) }{ P(\nu^{u} \geq 2|u \in U^{[y_1,y_2],g}) }$', transform=ax.transAxes, fontsize=14)
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)
ax.grid()
ax.set_xlim(-.6,len(ageinds)-0.4)
#ax.set_ylim(-0.03,0.43)

print 'Export Plot File'
#plt.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.92, wspace=0.20, hspace=0.20)
plt.tight_layout()
plt.savefig('images/img-rc-age-gender.pdf', dpi=300)
plt.close()



