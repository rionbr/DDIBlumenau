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
#
import numpy as np
import geopandas as gpd
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import util
import statsmodels.api as sm
from itertools import cycle


if __name__ == '__main__':

	# Init
	cmap = 'Blues'
	title = r'Population $P(\Omega^{N})$'

	# Load BNU
	dfBnu = util.BnuData(age_per_gender=False)
	print dfBnu.columns
	dfBnu['P(population)'] = dfBnu['population'] / dfBnu['population'].sum()

	print '--- Load .shp Files ---'
	gdf_bairros = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/DIVISAO_BAIRROS.shp').to_crs({'init': u'epsg:4674'})
	gdf_limites = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/LIMITE_MUNICIPAL.shp').to_crs({'init': u'epsg:4674'})
	gdf_hidro = gpd.GeoDataFrame.from_file(r'shapefiles_blumenau/HIDROGRAFIA_PRINCIPAL.shp').to_crs({'init': u'epsg:4674'})

	gdf_bairros['geometry'] = gdf_bairros['geometry'].simplify(tolerance=0.00001)
	gdf_limites['geometry'] = gdf_limites['geometry'].simplify(tolerance=0.0001)
	gdf_hidro['geometry'] = gdf_hidro['geometry'].simplify(tolerance=0.0001)
	
	gdf_bairros['income'] = gdf_bairros['BAIRROS'].map( dfBnu['avg_income'] )
	gdf_bairros['P(population)'] = gdf_bairros['BAIRROS'].map( dfBnu['P(population)'] )
	gdf_bairros['centroid'] = gdf_bairros['geometry'].centroid

	# Bairro Names
	newcols = {
		1:[u'Vila Itoupava','r',15],
		18:[u'Fidélis','r',14],
		9:[u'Fortaleza Alta','r',13],
		8:[u'Fortaleza','r',12],
		21:[u'Itoupava Norte','r',11],
		19:[u'Tribéss','r',10],
		28:[u'Itoupava Seca','r',9],
		4:[u'Nova Esperança','r',8],
		14:[u'Ponta Aguda','r',7],
		16:[u'Vorstadt','r',6],
		22:[u'Boa Vista','r',5],
		0:[u'Victor Konder','r',4],
		2:[u'Ribeirão Fresco','r',3],
		26:[u'Centro','r',2],
		10:[u'Garcia','r',1],
		15:[u'Da Glória','r',0],

		33:[u'Itoupava Central','l',18],
		32:[u'Itoupavazinha','l',17],
		23:[u'Salto do Norte','l',16],
		3:[u'Do Salto','l',15],
		7:[u'Vila Nova','l',14],
		31:[u'Testo Salto','l',13],
		20:[u'Escola Agrícola','l',12],
		27:[u'Badenfurt','l',11],
		34:[u'Salto Weisbach','l',10],
		13:[u'Água Verde','l',9],
		6:[u'Velha','l',8],
		5:[u'Passo Manso','l',7],
		25:[u'Bom Retiro','l',6],
		11:[u'Velha Central','l',5],
		29:[u'Jardim Blumenau','l',4],
		12:[u'Velha Grande','l',3],
		24:[u'Vila Formosa','l',2],
		30:[u'Valparaíso','l',1],
		17:[u'Progresso','l',0],
	}
	gdf_bairros[['NAME','lr','rank']] = pd.DataFrame(newcols.values(), columns=['NAME','lr','rank'])

	# COlor
	n = len(gdf_bairros)
	norm = mpl.colors.Normalize(vmin=0,vmax=n)
	colors = map(mpl.cm.get_cmap('tab10'), np.linspace(0,1,n, endpoint=False))

	#
	#
	#
	fig = plt.figure(figsize=(8,6))
	ax = plt.subplot(1, 1, 1)
	ax.set_aspect('equal')

	#
	# Blumenau
	#
	gdf_limites.plot(ax=ax, color='#fbeebf', linewidth=1.35, edgecolor='lightgray', zorder=1)
	# Bairro filling
	pb = gdf_bairros.plot(ax=ax, column='P(population)', cmap=cmap, linewidth=0, alpha=1, zorder=2)
	# Border only
	gdf_bairros.plot(ax=ax, color='none', facecolors='none', edgecolor='#464646', linewidth=0.5, alpha=1, zorder=4)
	# Rivers
	gdf_hidro.plot(ax=ax, linewidth=0.25, facecolor='#7ea6cb', edgecolor='#4169e1', zorder=3)
	# Bairro Centroid point
	gdf_bairros['geometry'].centroid.plot(ax=ax, c=colors, edgecolor='white', marker='o', markersize=38, linewidth=0.5, alpha=1, zorder=30)

	# Annotate
	xlims = ax.get_xlim()
	ymin = -27.00
	spacing = .015
	colors = cycle(['r', 'g', 'b', 'y'])
	#left_ys  = np.linspace(ylims[0], ylims[1], gdf_bairros.loc[gdf_bairros['lr']=='left' ,:].shape[0]+1)
	#right_ys = np.linspace(ylims[0], ylims[1], gdf_bairros.loc[gdf_bairros['lr']=='right',:].shape[0]+1)
	def annotate(x):
		i = int(x['rank'])
		if x['lr'] == 'l':
			x_text = xlims[0]+.026
			align = 'right'
			connectionstyle = "arc,angleA=0,armA=50,rad=10"
		else:
			x_text = xlims[1]-.026
			align = 'left'
			connectionstyle = "arc,angleA=180,armA=50,rad=10"
		y_text = ymin+i*spacing

		ax.annotate(
			s='',
			xy=x['centroid'].coords[0],
			xycoords='data',
			xytext=(x_text,y_text),
			textcoords='data',
			arrowprops=dict(
				arrowstyle="<|-,head_length=0.3,head_width=0.15",
				connectionstyle=connectionstyle,
				fc="k", ec="k",
				linewidth=.4),
			zorder=25)
		ax.annotate(
			s=x['NAME'],
			xy=x['centroid'].coords[0],
			xycoords='data',
			xytext=(x_text,y_text),
			textcoords='data',
			va='center',
			ha=align,
			zorder=25)

	gdf_bairros.apply(annotate, axis=1)

	ax.set_xlim(-49.21,-49.00)
	ax.set_ylim(-27.01,-26.70)

	vmin = gdf_bairros['P(population)'].min()
	vmax = gdf_bairros['P(population)'].max()

	# ColorBar
	#cax = fig.add_axes([0.36, 0.09, 0.22, 0.018]) # [left, bottom, width, height] 
	cax = fig.add_axes([0.70, 0.88, 0.23, 0.018]) # [left, bottom, width, height] 
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	sm._A = []
	fig.colorbar(sm, cax=cax, format='%.2f', ticks=np.linspace(vmin,vmax,3), orientation='horizontal')

	ax.text(x=1.135, y=0.92, s=title, transform=ax.transAxes, ha='center')

	# Hide Axis
	ax.set_frame_on(False)
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)

	print 'Saving File'
	#_ = ax.axis('off')
	plt.subplots_adjust(left=0.20, bottom=0.01, right=0.80, top=.99, wspace=0.0, hspace=0.0)
	plt.savefig('images/maps/img-map-blumenau.pdf', dpi=300, pad_inches=0.0)

