# coding=utf-8
# Author: Rion B Correia
# Date: Nov 29, 2018
#
# Description:
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
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8') # Null Models (computed by `compute_rri_null_models.py`)

# Sum Males + Females
dfN['u^{i}_{rnd}'] = dfN['u^{i,F}_{rnd}'] + dfN['u^{i,M}_{rnd}']

n_user = dfR['u'].sum() # This is only valid if the sample of users is 100%
n_runs = dfN['run'].max()

# Group all the Runs in the NullModel
dfN = dfN.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	('u^{i}_{rnd}',['mean','std']),
]))
dfN.columns = ['-'.join(col).strip() for col in dfN.columns.values]


dfR00_89 = dfR.iloc[ 0:18, : ]
dfR90_pl = dfR.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfR = pd.concat([dfR00_89, dfR90_pl], axis=0)

dfR['RC^{y}'] = (dfR['u^{c}'] / dfR['u']) / (dfR['u^{n2}'] / dfR['u'])

print '>> dfR'
print dfR

#
#
#
print '--- Plotting ---'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.3,3))
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)
plt.rc('legend', labelspacing=0.3)


width = 0.33
ms = 8
ms_t = 5
ms_b = 10
lw=1.2
ageinds = np.arange(0, dfR.shape[0])
agelabels = dfR.index.values

rc, = ax.plot(ageinds, dfR['RC^{y}'].values, color='orange', marker='o', lw=0, markersize=ms, zorder=5)

ax.axvspan(2.5, 6.5, alpha=0.35, color='gray')


#
#Curve Fitting
#
print '--- Curve Fitting ---'
y_rc = dfR['RC^{y}'].values
x = np.arange(len(y_rc))
# x_ is used to plot the curve after
x_ = np.linspace(x[0],x[-1],len(x)*10)


# RC Linear Model
print '> RC Linear Model'
Xc = sm.add_constant( x )
rc_l_model = sm.OLS(y_rc, Xc)
rc_l_model_result = rc_l_model.fit()
#print rc_l_model_result.summary()

# RC Cubic Model
print '> RC CUBIC'
Xc = sm.add_constant( np.column_stack( [x**3, x**2, x] ) )
rc_c_model = sm.OLS(y_rc, Xc)
rc_c_model_result = rc_c_model.fit()
print rc_c_model_result.summary()
Xc_ = sm.add_constant( np.column_stack( [x_**3, x_**2, x_] ) )
y_rc_ = np.dot(Xc_, rc_c_model_result.params)
rc_f_cubic = ax.plot(x_, y_rc_, color='orange', lw=2, zorder=3)
rc_c_model_R2 = rc_c_model_result.rsquared_adj
# ANOVA
anova_result = anova_lm(rc_l_model_result, rc_c_model_result)
print anova_result

ax.text(x=0.97, y=0.03, s=r'$R^2={:.3f}$'.format(rc_c_model_R2), ha='right',va='bottom', transform=ax.transAxes)


ax.legend([rc],[r'$RC^{[y_1,y_2]}$'],
	loc=2, handletextpad=0.5, columnspacing=0, handlelength=2, ncol=1)

ax.set_title(r'$RC^{[y_1,y_2]}$')
#ax.text(0, 1.09, r'$\frac{ P(\Psi^{u}>0|u \in U^{[y_1,y_2]}) }{ P(\nu^{u} \geq 2|u \in U^{[y_1,y_2]}) }$', transform=ax.transAxes, fontsize=14)
ax.set_xticks(ageinds)
ax.set_xticklabels(agelabels, rotation=90)
ax.grid()
ax.set_xlim(-.6,len(ageinds)-0.4)
#ax.set_ylim(-3,50)

print 'Export Plot File'
#plt.subplots_adjust(left=0.05, bottom=0.08, right=0.97, top=0.92, wspace=0.20, hspace=0.20)
plt.tight_layout()
plt.savefig('images/img-rc-age.pdf', dpi=300)
plt.close()



