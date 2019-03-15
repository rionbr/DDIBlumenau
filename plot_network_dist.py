# coding=utf-8
# Author: Rion B Correia
# Date: Nov 07, 2018
#
# Description: Plots Network Edge Distribution
#
#
from __future__ import division
import os
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.precision', 2)


if __name__ == '__main__':

	
	dfN = pd.read_csv('csv/net_nodes.csv', encoding='utf-8')
	dfE = pd.read_csv('csv/net_edges.csv', encoding='utf-8')

	print dfN.head()
	deg = dfN['degree'].sort_values(ascending=False).reset_index(drop=True)
	degstr = dfN['degree-strength'].sort_values(ascending=False).reset_index(drop=True)


	print dfE.head()
	tau = dfE['tau'].sort_values(ascending=False).reset_index(drop=True)
	u = dfE['patients'].sort_values(ascending=False).reset_index(drop=True)

	#
	#
	#
	fig, (ax1,ax2) = plt.subplots(figsize=(7,3.3), nrows=1, ncols=2)
	plt.rc('font', size=12)
	#plt.rc('legend', fontsize=10)
	plt.rc('legend', numpoints=3, labelspacing=.2, handletextpad=0.3, handlelength=1.2, borderpad=0.3)
	plt.rc('legend', )
	#
	# Node Dist
	#
	marker = 'o'
	lw = 0
	ms = 5

	p1, = ax1.plot(deg.index, deg.values, color='darkred', marker='o', lw=lw, ms=ms, zorder=5, label=r'$deg(i)$')
	p2, = ax1.plot(degstr.index, degstr.values, color='goldenrod', marker='D', lw=lw, ms=ms, zorder=5, label=r'$degstr(i)$')
	ax1.set_title(r'Nodes')
	ax1.axhline(1, color='black', lw=0.75)
	ax1.grid(True)
	ax1.set_ylim(.025,deg.max()+3)
	ax1.set_xlim(.8,100)
	ax1.set_yscale('log')
	ax1.set_xscale('log')
	ax1.set_ylabel('value')
	ax1.set_xlabel('rank')

	ax1.legend(loc='lower left', bbox_to_anchor=(0.,0.))

	#
	# Edge Dist
	#	
	p1, = ax2.plot(u.index, u.values, color='darkgreen', marker='D', lw=lw, markersize=ms, zorder=5, label=r'$U^{\Phi}_{i,j}$')
	p2, = ax2.plot(tau.index, tau.values, color='blueviolet', marker='o', lw=lw, markersize=ms, zorder=5, label=r'$\tau^{\Phi}_{i,j}$')
	ax2.axhline(1, color='black', lw=0.75)
	ax2.set_title(r'Edges')
	ax2.grid(True)
	ax2.set_ylim(.005,u.max()+0.1)
	ax2.set_xlim(.8,300)
	ax2.set_yscale('log')
	ax2.set_xscale('log')
	ax2.set_ylabel('weight')
	ax2.set_xlabel('rank')

	ax2.legend(loc='center left', bbox_to_anchor=(0.0,0.6))

	print 'Export Plot File'
	plt.subplots_adjust(left=0.10, bottom=0.16, right=0.98, top=0.90, wspace=0.35, hspace=0.25)
	#plt.tight_layout()
	plt.savefig('images/img-graph-dist.pdf', dpi=300)
	plt.close()