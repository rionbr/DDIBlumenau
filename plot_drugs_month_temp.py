# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot DDI timelines
#
#
# coding=utf-8
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import util
from datetime import datetime

# Plot Styles
styles = ['r-o','g-o','b-o','c-o','m-o',  'r-s','g-s','b-s','c-s','m-s',  'r^','g^','b^','c^','m^']
months = ['Jan','April','Jul','Oct','Jan','April','Jul']


#
# Load CSVs
#
df_file = 'data/dumpsql_final.csv'
df = pd.read_csv(df_file, encoding='utf-8', parse_dates=['date_disp'], nrows=None, dtype={'id_usuario':np.int64})
#dfu, dfc, dfi, dfs = util.dfUsersInteractionsSummary()

df['qt_drugs'] = 1
print df.head()


# Load Clima
dfClima = util.dfBnuClima()
dfClima = pd.concat([dfClima, dfClima , dfClima ])
print dfClima
dfClima['date'] = pd.date_range(start='2013-01-01', end='2015-12-31', freq='MS')
dfClima = dfClima.set_index('date')

print '>> dfClima'
print dfClima


#
# Plot Timelines of DDI
#
print '--- Grouping Month-Dispensed (Month) ---'
dfg = df.groupby(pd.Grouper(key='date_disp', freq='MS')).agg(
		{
			'qt_drugs':'sum'
		})

print dfg.head()

# Transform in Thousands
dfg['qt_drugs'] = dfg['qt_drugs'] / 1000.


# Remove
#dfsg = dfsg.loc[ ~dfsg.index.isin(['2015-07','2015-08']), : ]

#
# Plot
#
print '- Plotting -'
#fig = plt.figure(figsize=(10,4))
fig = plt.figure(figsize=(5.5,3))
ax = plt.subplot(1, 1 ,1)

plt.rc('font', size=12)
plt.rc('legend', fontsize=10)
plt.rc('legend', numpoints=1)


ax.set_title('Drug intervals dispensed')
ax.plot(dfg.loc[:,:].index , dfg.loc[:,'qt_drugs'].values, label='Dispensed', c='green', ls='-', marker='o', markersize=8, zorder=99)
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='both', which='minor', labelsize=0)
ax.grid(which='major')


ax.set_ylabel(r'$\alpha$ (in thousands)')

months_maj = MonthLocator(range(1, 13), bymonthday=1, interval=4)
months_min = MonthLocator(range(1, 13), bymonthday=1, interval=1)
months_maj_fmt = DateFormatter("%b %y")
ax.xaxis.set_major_locator(months_maj)
ax.xaxis.set_major_formatter(months_maj_fmt)
ax.xaxis.set_minor_locator(months_min)

ax.set_xlim(datetime(2013,12,15),datetime(2015,07,01))
#ax.set_ylim(50,115)



#
axb = ax.twinx()
axb.plot(dfClima.index.values, dfClima['temp_c_mean'].values, c='orange',ls='-', marker='', lw=4, alpha=0.6, zorder=5)
axb.fill_between(dfClima.index.values, dfClima['temp_c_min'].values, dfClima['temp_c_max'].values, facecolor='orange', linewidth=2, edgecolor='orange', alpha=.3, zorder=4)
axb.axvspan(datetime(2014,01,01), datetime(2014,06,30), facecolor='grey', alpha=0.3, zorder=1)
axb.set_ylabel('Temp. $^{\circ}$C')

axb.set_ylim(0,30)

axb.xaxis.set_major_locator(months_maj)
axb.xaxis.set_major_formatter(months_maj_fmt)

ax.set_zorder(axb.get_zorder()+1) #put ax in front of axb
ax.patch.set_visible(False) # hide the 'canvas' 


def lagged_corr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

print dfClima.loc[ '2014-06-01':'2015-06-30','temp_c_mean']
print dfg.loc['2014-06': , 'qt_drugs']

print 'AutoCorrelation:'
print 'Clima:' , dfClima.loc[ '2014-06-01':'2015-06-30','temp_c_mean'].autocorr(lag=1)
print 'QtDrugs:' , dfg.loc['2014-06': , 'qt_drugs'].autocorr(lag=1)

print 'Correlation:'
print dfClima.loc[ '2014-06-01':'2015-06-30','temp_c_mean'].corr(dfg.loc['2014-06':,'qt_drugs'])
print 'Lagged Correlation:'
print lagged_corr( dfClima.loc[ '2014-06-01':'2015-06-30','temp_c_mean'] , dfg.loc['2014-06':,'qt_drugs'] , lag=1)


print 'Export Plot File'
#plt.subplots_adjust(left=0.08, bottom=0.22, right=0.98, top=0.92, wspace=0.35, hspace=0.0)
plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.92, wspace=0.35, hspace=0.0)
plt.tight_layout()
plt.savefig('images/img-drugs-month-temp.pdf', dpi=300)
plt.close()


