# coding=utf-8
# Author: Rion B Correia
# Date: Nov 9, 2018
# 
# Description: 
# Plots PCA
#
import sys
sys.path.insert(0, '../../include')
#sys.path.insert(0, '../../../include')
#
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')

mpl.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
from adjustText import adjust_text
#
# Label Scatter Plot
#
def labelScatterPlot(labels, ax, xs, ys, n, stdn, printLabel):
	df = pd.DataFrame({'xs':xs,'ys':ys}, index=labels)
	stdx , stdy = df['xs'].std() , df['ys'].std()
	df = df.loc[ (df['xs'] >= stdx*stdn) | (df['xs'] <= -(stdx*stdn)) | (df['ys'] >= stdy*stdn) | (df['ys'] <= -(stdy*stdn)) ]

	df = df.sort_values('xs', ascending=False)
	textobjs = list()
	for label, data in df.iterrows():
		textobj = ax.text(data['xs'], data['ys'], label, fontsize=8, alpha=1, zorder=10, ha='center', va='center')
		textobjs.append(textobj)
	return textobjs
#
# Annotate Scatter Plot
#
def annotatePoint(label, ax, x, y, xpad, ypad):
	obj = ax.annotate(
		label, 
		xy = (x, y), xytext = (x, y),
		xycoords='data', textcoords='data', ha='center', va='center',
		bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.3),
		#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.5', alpha = 0.2),
		fontsize=9, alpha=0.65
		)
	return obj



def plot_pca(network):
	
	print '--- Plotting PCA : {:s} ---'.format(network)
	#
	# Files
	#
	rPCAFile = 'csv/net_pca_{:s}.csv'.format(network)
	rSFile = 'csv/net_pca_{:s}-s.csv'.format(network)
	wIMGFile = 'images/img-graph-pca-{:s}.pdf'.format(network)

	print '> Loading PCA .csv'
	dfPCA = pd.read_csv(rPCAFile, index_col=0, encoding='utf-8')
	s = pd.read_csv(rSFile, squeeze=True, index_col=0, header=None, encoding='utf-8')

	print dfPCA.shape
	print s.head()


	#
	# Plot PCA
	#
	print '> Plotting'


	if network == 'tau':
		title = r'PCA on $\tau^{\Phi}_{i,j}$ network'
	elif network == 'u':
		title = r'PCA on $U^{\Phi}_{i,j}$ network'

	
	
	fig, ((ax1,ax2,ax3),(ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(figsize=(12,10),  nrows=3, ncols=3)
	
	#fig.suptitle(title)
	#plt.rc('font', size=11)
	plt.rc('figure', titlesize=12)
	plt.rc('legend', fontsize=12)
	plt.rc('legend', scatterpoints=1)
	
	#
	# Plot EigenVals
	#
	s_cumsum = s.cumsum()
	n_eigen_95 = s_cumsum[s_cumsum<0.95].shape[0]

	n = 9
	ind = np.arange(n)
	height = s.iloc[:n].values
	width = 0.60
	xticklabels = ind+1

	cmap = mpl.cm.get_cmap('hsv_r')
	norm = mpl.colors.Normalize(vmin=0,vmax=n)
	s_colors = map(cmap, np.linspace(0,1,n, endpoint=False))

	ax1.bar(ind, height, width, color=s_colors, zorder=9, edgecolor='black', align='center')
	ax1.set_xticks(ind)
	ax1.set_xticklabels(xticklabels)

	ax1.set_title('Explained variance ratio')
	ax1.annotate('95% with {:d}\nsingular vectors'.format(n_eigen_95), xy=(0.97, 0.97), xycoords="axes fraction", ha='right', va='top')
	ax1.set_xlabel('Components')
	ax1.set_ylabel('%')
	
	ax1.grid()
	ax1.set_xlim(-.5,n-.5)
	
	
	for dim, ax in zip( range(1,10), [ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9] ):
		print '> Dim: {:d}'.format(dim,dim+1)
		col = str(dim)+'c'
		x = str(dim)+'c'
		y = str(dim+1)+'c'
		xs = dfPCA[x].tolist()
		ys = dfPCA[y].tolist()
		pca_colors = dfPCA['color'].tolist()

		#if pca_colors == None:
		#	pca_colors = ys
		#	norm = mpl.colors.Normalize( vmin=min(ys), vmax=max(ys) )
		#	ax.scatter(xs,ys, c=pca_colors, cmap=cmapR2G2G, norm=norm, marker='o', edgecolor='black', lw=0.5, s=25)
		#else:
		ax.scatter(xs, ys, c=pca_colors, marker='o', edgecolor='black', lw=0.5, s=30, zorder=5, rasterized=True)
		
		# Draw a X at the center
		#ax.plot(0,0, color='black', marker='x', ms=16)
		
		# Draw lines at the center
		ax.axhline(y=0, c='gray', lw=0.75, ls='-', zorder=2)
		ax.axvline(x=0, c='gray', lw=0.75, ls='-', zorder=2)

		ax.set_title('Components {} and {}'.format(dim, dim+1) )
		ax.set_xlabel('Component %d' % (dim))
		ax.set_ylabel('Component %d' % (dim+1))
		
		
		ax.grid()
		ax.axis('equal')
		ax.locator_params(axis='both', tight=True, nbins=6)
		
		#ax.set_aspect('equal')
		#xmin, xmax = ax.get_xlim()
		#ax.set_xticks(np.round(np.linspace(xmin, xmax, 6), 2))
		#ymin, ymax = ax.get_ylim()
		#ax.set_yticks(np.round(np.linspace(ymin, ymax, 6), 2))
		
		labels = labelScatterPlot(dfPCA['label'].values, ax, xs, ys, n=2, stdn=2, printLabel=True)
		adjust_text(
			labels, x=xs, y=ys, ax=ax,
			lim=1000,
			force_text=(.1, .4), force_points=(.1, .4), force_objects=(1, 1),
			expand_text=(1.4, 1.8), expand_points=(1.4, 1.8), expand_objects=(1,1), expand_align=(1.1,1.9),
			only_move={'points':'xy','text':'xy','objects':'xy'},
			text_from_points=True,
			ha='center',va='center', autoalign=False,
			arrowprops=dict(arrowstyle="->", shrinkB=5, color='gray', lw=0.5, connectionstyle='angle3'),
		)
		

	# Save
	plt.tight_layout()
	plt.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.95, wspace=0.32, hspace=0.35)
	plt.savefig(wIMGFile, dpi=150, bbox_inches=None, pad_inches=0.0)
	plt.close()


if __name__ == '__main__':

	plot_pca('tau')
	plot_pca('u')

