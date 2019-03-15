# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of Interaction on Map
#
#
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#
import numpy as np
import geopandas as gpd
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
import util
import statsmodels.api as sm


if __name__ == '__main__':


	dfu, dfc, dfi = util.dfUsersInteractionsSummary()
	
	print '>> dfu'
	print dfu.head()

	# Load BNU
	dfBnu = util.BnuData(age_per_gender=False)
	dfBnu['P(population)'] = dfBnu['population'] / dfBnu['population'].sum()
	print '>> dfBnu'
	print dfBnu[['population','gender_rate','avg_income','P(population)']].head()

	# Group by Bairro
	dfub = dfu.groupby('hood').agg({'n_a':'sum','n_i':'sum','n_ij':'sum','n_ij_ddi':'sum'})
	dfub['n_a_per_capita'] = dfub['n_a'] / dfBnu['population']
	dfub['n_i_per_capita'] = dfub['n_i'] / dfBnu['population']
	dfub['n_ij_per_capita'] = dfub['n_ij'] / dfBnu['population']
	dfub['n_ij_ddi_per_capita'] = dfub['n_ij_ddi'] / dfBnu['population']
	print '>> dfub'
	print dfub.head()

	column = 'n_ij_ddi_per_capita'

	legend_format = '%.2f'

	if column == 'P(population)':
		cmap = 'Blues'
		filesuffix = 'pop'
		title = r'Population $P(\Omega^{N})$'

	elif column == 'gender_rate':
		cmap = 'seismic'
		filesuffix = 'gender'
		title = r'Gender Rate $\Omega^{N,g=\mathrm{F}} / \Omega^{N,g=\mathrm{M}}$'

	if column == 'avg_income':
		cmap = 'Oranges'
		filesuffix = 'income'
		title = r'Average Income (R$/month)'
		legend_format = '%d'

	elif column == 'n_a_per_capita':
		cmap = 'Greens'
		filesuffix = 'intervals'
		title = r'Drug Intervals per capita $\alpha^{N}/\Omega^{N}$'

	elif column == 'n_i_per_capita':
		#cmap = LinearSegmentedColormap.from_list('Magentas', ['white', 'red'])
		cmap = 'Purples'
		filesuffix = 'drugs'
		title = r'Drugs per capita $\nu^{N}/\Omega^{N}$'

	elif column == 'n_ij_per_capita':
		cmap = 'Oranges'
		filesuffix = 'coadmin'
		title = r'Co-administrations per capita $\Psi^{N}/\Omega^{N}$'

	elif column == 'n_ij_ddi_per_capita':
		cmap = 'Reds'
		filesuffix = 'inter'
		title = r'Drug Interaction per capita $\Phi^{N}/\Omega^{N}$'


	print '--- Load .shp Files ---'
	gdf_bairros = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/DIVISAO_BAIRROS.shp').to_crs({'init': u'epsg:4674'})
	gdf_limites = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/LIMITE_MUNICIPAL.shp').to_crs({'init': u'epsg:4674'})
	gdf_hidro = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/HIDROGRAFIA_PRINCIPAL.shp').to_crs({'init': u'epsg:4674'})

	gdf_bairros['geometry'] = gdf_bairros['geometry'].simplify(tolerance=0.0001)
	gdf_limites['geometry'] = gdf_limites['geometry'].simplify(tolerance=0.001)
	gdf_hidro['geometry'] = gdf_hidro['geometry'].simplify(tolerance=0.0001)

	gdf_bairros['P(population)'] = gdf_bairros['BAIRROS'].map( dfBnu['P(population)'] )
	gdf_bairros['gender_rate'] = gdf_bairros['BAIRROS'].map( dfBnu['gender_rate'] )
	gdf_bairros['avg_income'] = gdf_bairros['BAIRROS'].map( dfBnu['avg_income'] )
	
	gdf_bairros['n_a_per_capita'] = gdf_bairros['BAIRROS'].map( dfub['n_a_per_capita'] )
	gdf_bairros['n_i_per_capita'] = gdf_bairros['BAIRROS'].map( dfub['n_i_per_capita'] )
	gdf_bairros['n_ij_per_capita'] = gdf_bairros['BAIRROS'].map( dfub['n_ij_per_capita'] )
	gdf_bairros['n_ij_ddi_per_capita'] = gdf_bairros['BAIRROS'].map( dfub['n_ij_ddi_per_capita'] )

	#
	# Plot
	#
	fig = plt.figure(figsize=(3.5,6))

	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')

	#
	# Blumenau
	#
	gdf_limites.plot(ax=ax, color='#fbeebf', linewidth=1.35, edgecolor='lightgray', zorder=1)
	# Bairro filling
	
	pb = gdf_bairros.plot(ax=ax, column=column, cmap=cmap, linewidth=0, alpha=1, zorder=2)
	# Border only
	gdf_bairros.plot(ax=ax, color='none', facecolors='none', edgecolor='#464646', linewidth=0.5, alpha=1, zorder=4)
	# Rivers
	gdf_hidro.plot(ax=ax, linewidth=0.25, facecolor='#7ea6cb', edgecolor='#4169e1', zorder=3)

	#ax.set_ylabel(ylabel)

	# Colorbar
	if column == 'gender_rate':
		vmin = 0.8
		vmax = 1.2
	else:
		vmin = gdf_bairros[column].min()
		vmax = gdf_bairros[column].max()

	cax = fig.add_axes([0.12, 0.58, 0.032, 0.30]) # [left, bottom, width, height] 
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	fig.colorbar(sm, cax=cax, format=legend_format, ticks=np.linspace(vmin,vmax,5), orientation='vertical')

	ax.set_title(title, fontsize=12)

	ax.set_xlim(-49.19,-49.003)
	ax.set_ylim(-27.01,-26.70)

	# Hide Axis
	#ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)

	print 'Saving File'
	#_ = ax.axis('off')
	plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=.95, wspace=0.0, hspace=0.0)
	plt.savefig('images/maps/img-map-blumenau-%s.pdf' % (filesuffix), dpi=300, pad_inches=0.0)

