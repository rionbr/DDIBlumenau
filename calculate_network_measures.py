# coding=utf-8
# Author: Rion B Correia
# Date: Nov 8, 2018
#
# Description: Loads DDI networks and calculates network measures
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
import community
from infomap import infomap
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.precision', 2)
#pd.set_option('display.float_format', lambda x: '%.2f' % x)
import util



def compute_network_measures(network):


	G = nx.read_gpickle('graphs/ddi_network_{:s}.gpickle'.format(network))


	print '--- G ---'
	# Sum edge value
	n_nodes = G.number_of_nodes()
	n_edges = G.number_of_edges()

	tau_edges_female = [d['tau'] for u,v,d in G.edges(data=True) if d['gender']=='Female']
	tau_edges_male = [d['tau'] for u,v,d in G.edges(data=True) if d['gender']=='Male']

	n_edges_female = len(tau_edges_female)
	n_edges_male = len(tau_edges_male)
	
	sum_tau_edges_female = sum(tau_edges_female)
	sum_tau_edges_male = sum(tau_edges_male)
	
	print 'Number of Nodes: {:d}'.format(n_nodes)
	print 'Number of Edges: {:d}'.format(n_edges)
	print '--- Edges RRI Male / Female ---'
	print 'Female len(RRI^F>1): {:,d} ({:.2%})'.format( n_edges_female , n_edges_female,n_edges )
	print 'Male len(RRI^M>1): {:,d} ({:.2%})'.format( n_edges_male , n_edges_male/n_edges)

	print 'Sum of tau for women: {:.2f}'.format( sum_tau_edges_female ) 
	print 'Sum of tau for men: {:.2f}'.format( sum_tau_edges_male )

	# Removing Hormone Drugs
	print '\n--- Gnh - Except Hormones ---'
	female_hormones = ['Ethinyl Estradiol','Estradiol','Norethisterone','Levonorgestrel','Estrogens Conj.']
	Gnh = G.copy()
	node2remove = [n for n,d in Gnh.nodes(data=True) if d['hormone']==True]
	Gnh.remove_nodes_from(node2remove)
	
	n_nodes = Gnh.number_of_nodes()
	n_edges = Gnh.number_of_edges()

	n_edges_nh = Gnh.number_of_edges()

	tau_edges_female_nh = [d['tau'] for u,v,d in Gnh.edges(data=True) if d['gender']=='Female']
	tau_edges_male_nh = [d['tau'] for u,v,d in Gnh.edges(data=True) if d['gender']=='Male']

	n_edges_female_nh = len(tau_edges_female_nh)
	n_edges_male_nh = len(tau_edges_male_nh)

	sum_tau_edges_female_nh = sum(tau_edges_female_nh)
	sum_tau_edges_male_nh = sum(tau_edges_male_nh)
	
	print 'Number of Nodes: {:d}'.format(n_nodes)
	print 'Number of Edges: {:d}'.format(n_edges)
	print '-- Edges (NO HORMONES) RRI Male / Female --'
	print 'Female len(RRI^F): {:,d} ({:.2%})'.format( n_edges_female_nh , n_edges_female_nh/n_edges_nh )
	print 'Male len(RRI^M): {:,d} ({:.2%})'.format( n_edges_male_nh , n_edges_male_nh/n_edges_nh )
	

	print 'Sum of tau for women: {:.2f}'.format( sum_tau_edges_female_nh ) 
	print 'Sum of tau for men: {:.2f}'.format( sum_tau_edges_male_nh )




	print '> (in)Degree'
	d = dict(nx.degree(G))
	nx.set_node_attributes(G, d, 'degree')

	print '> (in)Degree Strength'
	d = dict(nx.degree(G, weight='tau'))
	nx.set_node_attributes(G, d, 'degree-strength')

	print '> Attribute Assortativity Coefficient'
	print 'Value: {:.2f}'.format( nx.attribute_assortativity_coefficient(G,'class') )

	#print '> Degree Centrality'
	#dc = nx.degree_centrality(G)
	#nx.set_node_attributes(G, dc, 'degree_centrality')

	print '> Node Betweeness Centrality'
	bc = nx.betweenness_centrality(G, k=None, normalized=True, weight='tau')
	nx.set_node_attributes(G, bc, 'node_bet_cent')

	print '> Edge Betweeness Centrality'
	ebc = nx.edge_betweenness_centrality(G, k=None, normalized=True, weight='tau')
	nx.set_edge_attributes(G, ebc, 'edge_bet_cent')

	print '> Average Clustering (C)'
	C = nx.average_clustering(G, weight='tau')
	print 'Clustering (C): {:2f}'.format(C)

	print '> Community detection (Q) Louvain'
	best_partition_original = community.best_partition(G, weight='tau')
	nx.set_node_attributes(G, best_partition_original, 'module-louvain')
	Q = community.modularity(best_partition_original, G)
	print "Community (Q): {:2f}".format(Q)

	print '> InfoMap'
	infomapWrapper = infomap.Infomap("--two-level --undirected --silent")
	# Building Infomap network from a NetworkX graph...
	dto = {n:i for i,n in enumerate(G.nodes(),start=0)}
	dfrom = {i:n for n,i in dto.items()}
	for i,j,d in G.edges(data=True):
	    i = dto[i]
	    j = dto[j]
	    infomapWrapper.addLink(i,j,d['tau'])
	# Run!
	infomapWrapper.run();
	tree = infomapWrapper.tree
	# Dict of Results
	dM = {}
	for node in tree.leafIter():
	    i = dfrom[node.originalLeafIndex]
	    dM[i] = node.moduleIndex()
	nx.set_node_attributes(G, name='module-infomap', values=dM)



	# Nodes
	dfN = pd.DataFrame.from_dict({n:d for n,d in G.nodes(data=True)}, orient='index')
	dfN.rename_axis('dbi', axis='index', inplace=True)
	dfN.sort_values('degree-strength', ascending=False, inplace=True)
	dfN['rank_degree'] = dfN['degree'].rank(method='min', ascending=False).astype(int)
	dfN = dfN.round({'node_bet_cent':2})
	dict_dbi_label = dfN['label'].to_dict()
	print dfN.head()

	# Edges
	dfE = pd.DataFrame.from_dict({(u,v):d for u,v,d in G.edges(data=True)}, orient='index')
	dfE.rename_axis(['dbi','dbj'], axis='index', inplace=True)
	dfE.sort_values('edge_bet_cent', ascending=False, inplace=True)
	print dfE.head()
	dfE['label_i'] = dfE.index.get_level_values(level=0).map(dict_dbi_label)
	dfE['label_j'] = dfE.index.get_level_values(level=1).map(dict_dbi_label)
	dfE = dfE.round({'edge_bet_cent':2})
	print dfE.head()

	print '--- DDI-Network (Nodes) ---'
	cols = ['rank_degree','label','degree','degree-strength','node_bet_cent','module-louvain','module-infomap','p_node','class']
	#print dfN[cols].to_latex(escape=False, index=False)

	print '-- Clustering Louvain --'
	for cluster, dfl in dfN.groupby('module-louvain'):
		cols = ['label','degree','degree-strength','node_bet_cent','p_node','class']
		print '> Cluster: {:d}'.format(cluster)
		print dfl[cols].to_latex(escape=False, index=False)

	print '-- Clustering InfoMap --'
	for cluster, dfl in dfN.groupby('module-infomap'):
		cols = ['label','degree','degree-strength','node_bet_cent','p_node','class']
		print '> Cluster: {:d}'.format(cluster)
		print dfl[cols].to_latex(escape=False, index=False)

	print '--- DDI-Network (Edges) ---'
	cols = ['label_i','label_j','edge_bet_cent']
	print dfE[cols].to_latex(escape=False, index=False)

	print '> Export .CSV of nodes and Edgs'
	#dfN.to_csv('csv/net_nodes.csv', encoding='utf-8')
	#dfE.to_csv('csv/net_edges.csv', encoding='utf-8')

	print '> Export to Graphml'
	#nx.write_graphml(G, 'graphs/ddi_network_{:s}_cluster.graphml'.format(network))

if __name__ == '__main__':

	network = 'tau'
	compute_network_measures(network=network)


