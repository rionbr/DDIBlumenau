# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot Map Brasil with Blumenau annotated
#
#
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import geopandas as gpd
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)



if __name__ == '__main__':

	print '--- Load .shp Files ---'
	gdf_blumenau = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/LIMITE_MUNICIPAL.shp').to_crs({'init': u'epsg:4674'})
	gdf_br = gpd.GeoDataFrame.from_file(r'shapefiles_brasil/LIM_Pais_A.shp')
	gdf_estados = gpd.GeoDataFrame.from_file(r'shapefiles_brasil/LIM_Unidade_Federacao_A.shp')

	gdf_blumenau['geometry'] = gdf_blumenau['geometry'].simplify(tolerance=0.01)
	gdf_br['geometry'] = gdf_br['geometry'].simplify(tolerance=0.01)
	gdf_estados['geometry'] = gdf_estados['geometry'].simplify(tolerance=0.01)

	
	# Select only Brasil
	gdf_br = gdf_br.loc[ (gdf_br['NOME']=='Brasil') , :]

	# Color SC differently
	gdf_estados['SC'] = gdf_estados['SIGLA'].apply(lambda x: 1 if x=='SC' else 0)
	print gdf_estados

	#
	# Plotting
	#
	fig = plt.figure(figsize=(6,6),)
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')

	dcmap = ListedColormap(['#F8DE7F','#f8a27f'])
	print repr(dcmap)
	#
	# Blumenau
	#
	gdf_blumenau.plot(ax=ax, color='#F8DE7F', linewidth=1.35, edgecolor='lightgray', zorder=1)

	#
	# Brasil
	#
	gdf_br.plot(ax=ax, facecolor='None', linewidth=1.55, edgecolor='lightgray', zorder=5)	
	gdf_estados.plot(ax=ax, column='SC', cmap=dcmap, categorical=True, edgecolor='#464646', linewidth=.75, alpha=1, zorder=4)
	gdf_blumenau.plot(ax=ax, color='#FF0000', linewidth=1.35, edgecolor='#990000', zorder=5)

	(x,y) = gdf_blumenau['geometry'].centroid[0].coords[0]
	ax.annotate(
		s='Blumenau',
		xy=(x+0.2,y-0.4),
		xycoords='data',
		xytext=(x-0.1,y-6),
		textcoords='data',
		ha='left',
		arrowprops=dict(
				arrowstyle="simple",
				fc="k", ec="k",
				linewidth=.4),
		zorder=25)
	ax.set_xlim(-75,-34)
	ax.set_ylim(-35,6)

	
	# Hide Axis
	ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	
	print 'Saving File'
	plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)
	#plt.tight_layout()
	plt.savefig('images/maps/img-map-brasil.pdf', dpi=300, pad_inches=0.0)
