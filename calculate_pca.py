# coding=utf-8
# Author: Rion B Correia
# Date: Nov 8, 2018
#
# Description: Compute MetricBackbone on DDI Network
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
# Sci-kit Learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




def calculate_pca(network):


	G = nx.read_gpickle('graphs/ddi_network_{:s}.gpickle'.format(network))

	wPCAFile = 'csv/net_pca_{:s}.csv'.format(network)
	wSFile   = 'csv/net_pca_{:s}-s.csv'.format(network)

	dict_label = nx.get_node_attributes(G, 'label')
	dict_class = nx.get_node_attributes(G, 'class')
	dict_color = nx.get_node_attributes(G, 'color')
	dict_pi = nx.get_node_attributes(G, 'p_node')
	dfX = nx.to_pandas_adjacency(G, weight=network)

	X = dfX.values

	print '> Normalizing'
	X = StandardScaler().fit_transform(X)

	print '> PCA'
	pca = PCA(n_components=None)
	res = pca.fit_transform(X)

	print '> to DataFrame'
	dfPCA = pd.DataFrame(res[:,0:9], columns=['1c','2c','3c','4c','5c','6c','7c','8c','9c'], index=dfX.index)
	dfPCA['label'] = dfX.index.map(lambda x: dict_label[x])
	dfPCA['class'] = dfX.index.map(lambda x: dict_class[x])
	dfPCA['color'] = dfX.index.map(lambda x: dict_color[x])
	dfPCA['pi'] = dfX.index.map(lambda x: dict_pi[x])

	s = pd.Series(pca.explained_variance_ratio_, index=range(1,res.shape[1]+1))


	print '> Saving to .CSV '
	dfPCA.to_csv(wPCAFile, encoding='utf-8')
	s.to_csv(wSFile, encoding='utf-8')

	print 'Done.'


if __name__ == '__main__':

	calculate_pca('tau')
	calculate_pca('u')