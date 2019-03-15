# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of Interaction on Map
#
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import util
import statsmodels.api as sm


#
# Load CSVs
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=False)


# Load BNU
dfBnu = util.BnuData(age_per_gender=False)

#dfCenso = util.dfCenso(age_per_gender=True)


# Add RendaMedia to every individual
#df['renda_media'] = df['bairro'].map(dfBnu['renda_media']) 
#df['nr_saude_total'] =  df['bairro'].map(dfBnu['nr_saude_total']) 

print '> dfBnu'
print dfBnu.head()
print '> dfu'
print dfu.head()

#
# Group Data
#
print '--- Grouping Data ---'
dfh = pd.concat([
	dfu.reset_index().groupby('hood').agg({
			'n_a':'sum',
			'id_user':pd.Series.nunique
		}).rename(columns={'id_user':'u'}),
	dfu.loc[ (dfu['n_ij_ddi']>0) , :].reset_index().groupby('hood', sort=False).agg({
			'n_ij':'sum',
			'id_user':pd.Series.nunique
		}).rename(columns={'id_user':'u^{i}','n_ij':'n_ij_ddi'})
	], axis=1)

print dfh.head()

# Add BnuDados 
dfh[['population','avg_income','n_health_total']] = dfBnu[['population','avg_income','n_health_total']]

# Normalize
dfh['n_a_pc'] = dfh['n_a'] / dfh['population']
dfh['n_ij_ddi_pc'] = dfh['n_ij_ddi'] / dfh['population']

# Remove OTHER
dfh = dfh.loc[ (dfh.index != 'OTHER') , : ]

# Sort
dfh = dfh.sort_values([('population')], ascending=False)

print '> dfh'
print dfh



cmM = LinearSegmentedColormap.from_list('myCmap',['red','blue','yellow','gold','goldenrod'])

#
# Plots
#
plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)

def textformat(value, tick_number):
	return "{:,.0f}".format(value)

#
# ax1
#
print '- Plotting -'
fig = plt.figure(figsize=(4.2,3.2))
ax1 = fig.add_subplot(1,1,1, adjustable='box')

cbticks = [800, 1200, 1600, 2000, 2400, 2800, 3200]
#
# Qt Drugs vs Population
#
x = dfh['population']
y = dfh['n_a']
c = dfh['avg_income']
s = 34

#OLS
X = sm.add_constant(x.values)
ols = sm.OLS(y.values, X).fit()
dfh['resid1'] = ols.resid
m, b = ols.params[1], ols.params[0]
points = np.linspace(x.min(), x.max(), 100)
ax1.plot(points, m*points+b, color='green', lw=1.5, zorder=0)

sc1 = ax1.scatter(x ,y, s=s, c=c, cmap=cmM, linewidths=0.5)


es = [
	# Below
	Ellipse(xy=(19200,42000), width=7000, height=20000, angle=-5, linewidth=1, fill=False, edgecolor='cyan'),
	# Above 1
	Ellipse(xy=(9324,83500), width=2500, height=10000, angle=-5, linewidth=1, fill=False, edgecolor='magenta'),
	# Above 2
	Ellipse(xy=(14471,127000), width=2500, height=10000, angle=-5, linewidth=1, fill=False, edgecolor='magenta')
]
for e in es:
	ax1.add_artist(e)
ax1.annotate('Fortaleza', xy=(14471, 126630), xycoords='data', xytext=(6000, 96630), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=0.3'), horizontalalignment='center', fontsize=10, style='italic')

ax1.text(0.02, 0.97, '$R^2=%.3f$\n$p=%.3f$' % (ols.rsquared,ols.pvalues[1]), ha="left", va="top", transform=ax1.transAxes)
ax1.set_title('Dispensation vs population')
ax1.set_ylabel(r'$\alpha^{N}$')
ax1.set_xlabel(r'$\Omega^{N}$')

n_d_ticksrange = np.arange(0, y.max()+15000, 30000)
n_d_tickslabels = [str(int(v/1000.))+'k' for v in n_d_ticksrange]
n_p_ticksrange = np.arange(0, x.max()+5000, 5000)
n_p_tickslabels = [str(int(v/1000.))+'k' for v in n_p_ticksrange]

ax1.set_xticks(n_p_ticksrange)
ax1.set_xticklabels(n_p_tickslabels)
ax1.set_yticks(n_d_ticksrange)
ax1.set_yticklabels(n_d_tickslabels)

ax1.set_xlim(-1000, max(n_p_ticksrange))
ax1.set_ylim(-5000, max(n_d_ticksrange)+30000)
ax1.grid()

cax1 = make_axes_locatable(ax1).append_axes('right',size='3.5%', pad=0.05)
cb1 = fig.colorbar(sc1, cax=cax1, format=plt.FuncFormatter(textformat), ticks=cbticks)
cax1.set_ylabel(r'R$')

plt.tight_layout()
plt.subplots_adjust(left=0.20, bottom=0.16, right=0.79, top=0.90, wspace=0.0, hspace=0.00)
plt.savefig('images/img-hood-disp.pdf', dpi=300, pad_inches=0.0)
plt.close()


#
# Qt Drugs/Population vs Qt Interaction/Population
#
print '- Plotting -'
fig = plt.figure(figsize=(4.2,3.2))
ax2 = fig.add_subplot(1,1,1, adjustable='box')


x = dfh['n_a_pc']
y = dfh['n_ij_ddi_pc']
c = dfh['avg_income']

#OLS
X = sm.add_constant(x.values)
ols = sm.OLS(y.values, X).fit()
dfh['resid2'] = ols.resid
m, b = ols.params[1], ols.params[0]
points = np.linspace(x.min(), x.max(), 100)
ax2.plot(points, m*points+b, color='green', lw=1.5, zorder=0)

sc2 = ax2.scatter(x, y, s=s, c=c, cmap=cmM, linewidths=0.5)

#ax2.annotate('Bom Retiro', xy=(3.601361, 0.457160), xycoords='data', xytext=(1.9, 0.54), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle='arc3,rad=0.3'), horizontalalignment='center', fontsize=11, style='italic')

ax2.text(0.02, 0.97, '$R^2=%.3f$\n$p=%.3f$' % (ols.rsquared,ols.pvalues[1]), ha="left", va="top", transform=ax2.transAxes)
ax2.set_title('Dispensation vs DDI')
ax2.set_xlabel(r'$\alpha^{N} / \Omega^{N}$')
ax2.set_ylabel(r'$\Phi^{N} / \Omega^{N}$')

ax2.set_xlim(0.0, 10)
ax2.set_ylim(0.0, 3.3)
ax2.grid()

cax2 = make_axes_locatable(ax2).append_axes('right',size='3.5%', pad=0.05)
cb2 = fig.colorbar(sc2, cax=cax2, format=plt.FuncFormatter(textformat), ticks=cbticks)
cax2.set_ylabel(r'R$')

plt.tight_layout()
plt.subplots_adjust(left=0.17, bottom=0.17, right=0.79, top=0.90, wspace=0.0, hspace=0.00)
plt.savefig('images/img-hood-ddi.pdf', dpi=300, pad_inches=0.0)
plt.close()


