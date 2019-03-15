# coding=utf-8
# Author: Rion B Correia
# Date: Nov 8, 2018
#
# Description: Build the DDI Networks (tau & u)
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
import networkx as nx
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.precision', 2)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)
import util


#
# Load CSVs
#
dfi = pd.read_csv('csv/ddi.csv', index_col=0, encoding='utf-8')
dfd = pd.read_csv('csv/drug.csv', index_col=0, encoding='utf-8')


print '-- dfi'
print dfi.head()
print '-- dfd'
print dfd.head()

dfi = dfi.round({'tau':2})
dfd = dfd.round({'P(i&j_ddi)':2})
dict_p_node = dfd['P(i&j_ddi)'].to_dict()

female_hormones = ['Ethinyl Estradiol','Estradiol','Norethisterone','Levonorgestrel','Estrogens Conj.']

# Set INF values to the MAX
maxrrivalue = dfi[['RRI^F','RRI^M']].replace([np.inf, -np.inf], np.nan).max().max().round(2)

dfi.loc[ dfi['RRI^F']==np.inf , 'RRI^F'] = maxrrivalue
dfi.loc[ dfi['RRI^M']==np.inf , 'RRI^M'] = maxrrivalue

# ReScale Weight Values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1,15))
dfi['u_ij_norm'] = scaler.fit_transform(dfi['u_ij'].values.reshape(-1,1))
scaler = MinMaxScaler(feature_range=(1,10))
dfi['tau_norm'] = scaler.fit_transform(dfi['tau'].values.reshape(-1,1))

#
# RRI Color (same code used to plot colorbar)
#
n = int(256 / 32)
bounds = np.arange(0,6,1)
boundaries = np.linspace(0.6,5,n*32).tolist()[2:-2] + [5.2,5.6]
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



# Color
dict_color = {
	'Cardiovascular agents':'#ee262c',
	'CNS agents':'#976fb0',
	'Hormones':'#f2ea25',
	'Anti-infectives':'#66c059',
	'Psychotherapeutic agents':'#c059a2',
	'Metabolic agents':'#f47a2b',
	'Respiratory agents':'#4da9df',
	'Gastrointestinal agents':'#6c8b37',
	'Nutritional Products':'#b4e2ee',
	'Topical Agents':'#ffe5cc',
	'Coagulation modifiers':'#f498b7'
	}

#
# Build Network
#
G = nx.Graph(name="DDI Network")

for i,r in dfi.iterrows():
	# Node
	db_i, db_j = r['db_i'], r['db_j']
	en_i, en_j = r['en_i'], r['en_j']
	class_i, class_j = r['class_i'], r['class_j']
	n_i, n_j = r['n_i'], r['n_j']
	len_i, len_j = r['len_i'], r['len_j']
	#
	hormone_i = True if en_i in female_hormones else False
	hormone_j = True if en_j in female_hormones else False

	# Female
	if r['RRI^F'] > 1.0:
		gender = 'Female'
		rri = r['RRI^F']
		rgbc = cmapF(norm(rri))
	else:
		gender = 'Male'
		rri = r['RRI^F']
		rgbc = cmapF(norm(rri))

		
	HEXc = mpl.colors.rgb2hex(rgbc)
	

	if not G.has_node(db_i):
		color_i = dict_color[class_i]
		p_node = dict_p_node[db_i]
		G.add_node(db_i, **{'label':en_i, 'class':class_i, 'len_i':len_i, 'color':color_i,'p_node':p_node,'hormone':hormone_i})
	if not G.has_node(db_j):
		color_j = dict_color[class_j]
		p_node = dict_p_node[db_j]
		G.add_node(db_j, **{'label':en_j, 'class':class_j, 'len_i':len_j, 'color':color_j,'p_node':p_node,'hormone':hormone_j})
	
	# Edges
	G.add_edge(db_i, db_j, **{
			'RRI^F'        :r['RRI^F'],
			'RRI^M'        :r['RRI^M'],
			'severity'     :r['severity'],
			'patients'     :r['u_ij'],
			'patients_norm':r['u_ij_norm'],
			#
			'tau'          :r['tau'],
			'tau_norm'     :r['tau_norm'],
			#
			'color'        :HEXc,
			'gender'       :gender
		}
	)

# G_tau and G_u
G_u   = G.copy()
G_tau = G.copy()

nx.set_edge_attributes(G_u, nx.get_edge_attributes(G_u, 'patients_norm'), 'weight')
nx.set_edge_attributes(G_tau, nx.get_edge_attributes(G_u, 'tau_norm'), 'weight')

## DEBUG

#for n,d in G.nodes(data=True):
#	print n,d

#for u,v,d in G.edges(data=True):
#	print u,v,d

print '- To GraphML'
#nx.write_graphml(G_u, 'graphs/ddi_network_u.graphml')
#
nx.write_gpickle(G_tau, 'graphs/ddi_network_tau.gpickle')
nx.write_gpickle(G_u, 'graphs/ddi_network_u.gpickle')



