# coding=utf-8
# Author: Rion B Correia
# Date: Nov 07, 2018
#
# Description: Plots a colorbar used in network
#
#
# coding=utf-8
from __future__ import division
import os
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
# Custom FONts
from matplotlib import font_manager

#font_list = font_manager.createFontList(['/Users/rionbr/Library/Fonts/BentonSansComp-Regular.otf'])
#font_manager.fontManager.ttflist.extend(font_list)
#print font_list
#print font_manager.fontManager.ttflist
path = '/Users/rionbr/Library/Fonts/BentonSansComp-Regular.otf'
prop = font_manager.FontProperties(fname=path)


#
# Init
#
n = int(256 / 32)
bounds = np.arange(0,6,1)
boundaries = np.linspace(0.6,5,n*32).tolist()[2:-2] + [5.2,5.6]

#
# Plot ColorBar
#
fig = plt.figure(figsize=(2, 2))
plt.rc('font', size=12, family=prop.get_name(), weight='regular')
plt.rc('legend', fontsize=10, numpoints=1, labelspacing=0.3, frameon=False)
plt.rc('axes', edgecolor='d8d9d8', linewidth=1)


axF = fig.add_axes([0.1, 0.05, 0.11, 0.9])
axM = fig.add_axes([0.6, 0.05, 0.11, 0.9])

reds = mpl.cm.Reds(np.linspace(0.2,0.8,n*12))
grays = np.array([mpl.colors.to_rgba('gray')] * n*3 )
blues = mpl.cm.Blues(np.linspace(0.2,0.8,n*12))

colorsM = np.vstack((grays, blues))
colorsF = np.vstack((grays, reds))


cmapM = mpl.colors.LinearSegmentedColormap.from_list('nx', colors=colorsM)
cmapF = mpl.colors.LinearSegmentedColormap.from_list('nx', colors=colorsF)

cmapM.set_over(blues[-1]) #'yellow')
cmapM.set_under('gray') #'gray')
cmapF.set_over(reds[-1]) #'yellow')
cmapF.set_under('gray')


norm = mpl.colors.Normalize(vmin=0, vmax=5)

cbF = mpl.colorbar.ColorbarBase(axF, cmap=cmapF, norm=norm, boundaries=boundaries, extend='max', extendfrac='auto', ticks=bounds, spacing='proportional', orientation='vertical')
cbM = mpl.colorbar.ColorbarBase(axM, cmap=cmapM, norm=norm, boundaries=boundaries, extend='max', extendfrac='auto', ticks=bounds, spacing='proportional', orientation='vertical')
#cb.set_label(r'$RRI^{F}$')

#
# Save Colorbar
#
plt.savefig('images/graphs/img-graph-colorbar-g.png', dpi=300, transparent=False)
plt.savefig('images/graphs/img-graph-colorbar-g.pdf', dpi=300, transparent=True)
plt.close()