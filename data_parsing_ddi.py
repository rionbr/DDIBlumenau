# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Parse the data and retrieve the data it is interacting with
#
#
# coding=utf-8
import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 300)
pd.set_option('display.width', 300)
import util
from datetime import datetime, timedelta
from itertools import combinations
import gzip
#
from time import sleep
from multiprocessing import Pool, Manager, cpu_count

print '--- Loading Dispensacao File ---'
# Load Dispensacao
df_file = 'data/dumpsql_final.csv'
df = pd.read_csv(df_file, parse_dates=['date_disp','dob'], nrows=None, dtype={'id_user':np.int64}, encoding='utf-8')
# Define Duracao TimeDelta
df['length_timedelta'] = pd.to_timedelta(df['length_days'], unit='D')
# Define Data Final da Duracao do Medicamento
df['date_end'] = df['date_disp'] + df['length_timedelta']
# Sort by Data Dispensacao
df.sort_values('date_disp', inplace=True)

# Remove Outsiders (Not from Blumenau)
df = df.loc[ (df['id_city']==4347) , : ]



# Load Dicts
print '--- Loading Translation File ---'
dfTra = pd.read_csv('dictionaries/drug_translation.csv', sep=',')
dfTra = dfTra.drop('problem', 1)
dfTra = dfTra.dropna()
#
print '--- Loading Drug Interaction File ---'
dfInt = pd.read_csv('dictionaries/drug_interaction.csv', header=0, index_col=0)


#
# Create Interaction HASH Dict
#
print '- Create Interaction Hash Dict'
dict_interaction = {}
for id, row in dfInt.iterrows():
	# DB Version ?
	id1,id2 = row['label'].split(' ')
	label = id1 + '-' + id2
	# DB Version 5
	#label = row['id1'] + '-' + row['id2']
	##
	dict_interaction[label] = row['interaction']

#
# Create Translation HASH Dict
#
print '- Create Translation Hash Dict'
dict_pronto_drugbank = {}
dict_drugbank_enpt = {}
for id, row in dfTra.iterrows():
	
	label = row['drugbank_label']

	drug_en = row['drug_en']
	drug_pt = row['drug_pt']
	drug_pronto = row['drug_pronto']
	#
	dict_pronto_drugbank[drug_pronto] = label
	#
	if '+' in label:
		label = label.split('+')
		drug_en = drug_en.split('+')
		drug_pt = drug_pt.split('+')
	else:
		label = [label]
		drug_en = [drug_en]
		drug_pt = [drug_pt]

	for (l, en, pt) in zip(label, drug_en, drug_pt):
		dict_drugbank_enpt[l] = [en, pt]



def db_label_map(x):
	try:
		return dict_pronto_drugbank[x]
	except:
		return ''

df['DB_label'] = df['drug_name'].apply(lambda x: db_label_map(x))

#
# Add new rows for composed drugs (ie: Amoxilina+Clavulanato)
#

print '--- Adding Rows for Composed Drugs ---'
dftmp = df.loc[ (df['DB_label'].str.contains('\+')) , :]
ls_newrows = []
i = 1
for j, (ind, row) in enumerate(dftmp.iterrows(), start=1):
	if j%10==0:
		print 'Iter: %d of %d' % (j, dftmp.shape[0])
	drug1name, drug2name = row['drug_name'].split('+')
	drug1name = drug1name.strip()
	drug2name = drug2name.strip()
	DB_label1, DB_label2 = row['DB_label'].split('+')
	old_row = row.copy(deep=True)
	new_row = row.copy(deep=True)
	
	old_row['drug_name'] = drug1name
	old_row['DB_label'] = DB_label1

	new_row['drug_name'] = drug2name
	new_row['DB_label'] = DB_label2

	ls_newrows.append(old_row)
	ls_newrows.append(new_row)
print '-- Creating New Rows DataFrame --'
df_newrows = pd.DataFrame(ls_newrows)
print '-- Dropping old rows ---'
df = df.loc[ ~(df['DB_label'].str.contains('\+')) , :]
print '-- Concatenating Dataframes --'
df = pd.concat([df,df_newrows], ignore_index=True)
df.sort_values(['date_disp','id_disp'], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
# Remove Drugs unmatched to DrugBank
df = df.loc[ df['DB_label'].str.startswith('DB') , : ]


#
# Users DataSet
#
dfUsers = df.copy(deep=True)
dfUsers = dfUsers.groupby(['id_user']).agg({
	'gender':'first',
	'marital':'first',
	'education':'first',
	'hood':'first',
	'dob':'first',
	'age':'first',
	})

#
# Qt Drugs to User
#
df_n_drugs = df.groupby('id_user').agg({'DB_label':['count',pd.Series.nunique]})
df_n_drugs.columns = df_n_drugs.columns.droplevel()
df_n_drugs.rename(columns={'count':'n_a','nunique':'n_i'}, inplace=True)
#
dfUsers['n_a'] = df_n_drugs['n_a'].fillna(0).astype(int)
dfUsers['n_i'] = df_n_drugs['n_i'].fillna(0).astype(int)

#
# Drugs Dispensed
#
dfDrugs = df.groupby(['id_user','DB_label']).agg({'id_disp':'count'}).rename(columns={'id_disp':'n_disp'})
dfDrugs.reset_index(inplace=True)
dfDrugs['en_i'] = dfDrugs['DB_label'].map(lambda x: dict_drugbank_enpt[x][0])

#dfDrugs['dpt'] = dfDrugs['DB_label'].map(lambda x: dict_drugbank_enpt[x][1])

#
# Loop All Users for Interactions
#
print '--- Looping all Users for Interactions ---'
#
dfG = df.groupby('id_user')

#
# Parallel Version
#
def worker(data):
	id_user, dfU, queue = data
	#print '--- %s ---' % (id_user)
	if len(dfU) == 0:
		return []
	else:
		result_user_coadmin_inter = []
		pairs = dfU['DB_label'].unique()

		for db_i,db_j in combinations(pairs, 2):
			#print '> %s - %s' % (db_i,db_j)
			
			# Order Drug Names Alphabetically
			if db_i > db_j:
				db_i, db_j = db_j, db_i

			dfi = dfU.loc[ dfU['DB_label']==db_i, : ]
			dfj = dfU.loc[ dfU['DB_label']==db_j, : ] 

			dfi = dfi.sort_values(['date_disp','date_end'], ascending=[True, False])
			dfj = dfj.sort_values(['date_disp','date_end'], ascending=[True, False])

			"""
			print '--- After Sorting ---'
			print dfi
			print dfj
			print '--- ---'
			def adjust_dates(r):
				global last_covered
				print '--- adjust_datas -'
				start, end = r['date_disp'] , r['date_end']
				print 'ID: %s' % r['id_disp']
				print 'Start: %s End: %s' % (start.date(), end.date())
				print 'Last: %s' % (last_covered)
				if last_covered is not None:
					if start < last_covered:
						if end < last_covered:
							return pd.Series([None,None])
						else:
							start = last_covered
				last_covered = end + timedelta(days=1)
				print 'Last: %s' % (last_covered)
				print 'Start: %s End: %s' % (start.date(), end.date())
				return pd.Series([start,end])

			# Adjust Time
			global last_covered
			last_covered = None
			print '--- Adjusting ---'
			print '-- dfi --'
			dfi[['date_disp_adj','date_end_adj']] = dfi[['id_disp','date_disp','date_end']].apply(adjust_dates, axis='columns')
			last_covered = None
			print '-- dfj --'
			dfj[['date_disp_adj','date_end_adj']] = dfj[['id_disp','date_disp','date_end']].apply(adjust_dates, axis='columns')

			print '--- After Adjust ---'
			print '-- dfi --'
			print dfi
			print '-- dfj --'
			print dfj
			
			dfi = dfi.loc[ ~(dfi['date_disp_adj'].isnull() | (dfi['date_end'].isnull())) , :]
			dfj = dfj.loc[ ~(dfj['date_disp_adj'].isnull() | (dfj['date_end'].isnull())) , :]
			
			#asd
			"""
			n_i = dfi.shape[0]
			n_j = dfj.shape[0]
			dfMi = pd.DataFrame.from_dict(
				{
					'i-%s'%i: {
						t : 1 for t in pd.date_range( r['date_disp'] , r['date_end'] ).tolist()
					} for i,r in dfi.iterrows()
				}).sum(axis=1).rename(db_i)
			dfMj = pd.DataFrame.from_dict(
				{
					'j-%s'%i: {
						t : 1 for t in pd.date_range( r['date_disp'] , r['date_end'] ).tolist()
					} for i,r in dfj.iterrows()
				}).sum(axis=1).rename(db_j)

			#dfM = pd.merge(dfMi, dfMj, how='inner', left_index=True, right_index=True).dropna(axis='index',how='all').dropna(axis='columns',how='all')
			dfM = pd.concat([dfMi, dfMj], axis=1).dropna(axis='index', how='all')
			
			#pd.set_option('display.max_rows', 600)
			#print dfM.head()

			counts = dfM.sum(axis=0).astype(np.int64).to_dict()

			len_i = counts[db_i]
			len_j = counts[db_j]

			len_ij = dfM.dropna(axis='index', how='any').shape[0]

			# No overlap
			if len_ij == 0:
				continue

			# Translate DrugBank Label to Languages
			en_i, dipt = dict_drugbank_enpt[ db_i ]
			en_j, djpt = dict_drugbank_enpt[ db_j ]

			# Check if there is an interaction
			try:
				text = dict_interaction[ db_i + '-' + db_j ]
			except:
				inter = False
				text = None
			else:
				inter = True
				text = dict_interaction[ db_i + '-' + db_j ]

			result_user_coadmin_inter.append(
				[id_user, db_i, en_i, n_i, len_i, db_j, en_j, n_j, len_j, len_ij, inter, text]
				)

		#end for (pairs)
		
		queue.put(id_user)

		return result_user_coadmin_inter
#
# Multiprocessing
#
n_cpu = cpu_count()
print '--- Starting Multiprocess (#cpu: %d) ---' % (n_cpu)
pool = Pool(n_cpu)
manager = Manager()
queue = manager.Queue()


dfGList = [(id_user, group, queue) for id_user, group in dfG]

run = pool.map_async(worker, dfGList)
while True:
	if run.ready():
		break
	else:
		size = queue.qsize()
		print 'Process: %d of %d completed' % (size , len(dfGList)-1)
		sleep(5)

result_list = run.get()
result_user_coadmin_inter = [record_list for a in result_list for record_list in a]

#
# Results to DataFrame
#
dfCoadmin = pd.DataFrame(result_user_coadmin_inter, columns=['id_user','db_i','en_i','n_i','len_i','db_j','en_j','n_j','len_j','len_ij','inter','text'])


# Add CoADmin and Inter to the User
dfCoadmin['db_ij'] = dfCoadmin['db_i'].str.cat(dfCoadmin['db_j'], sep='-')
df_coadmin = dfCoadmin.groupby('id_user').agg({'len_ij':'sum','db_ij':pd.Series.nunique})
df_coadmin.rename(columns={'len_ij':'len_ij','db_ij':'n_ij'}, inplace=True)
df_inter = dfCoadmin.loc[(dfCoadmin['inter']==True),:].groupby('id_user').agg({'len_ij':'sum','db_ij':pd.Series.nunique})
df_inter.rename(columns={'len_ij':'len_ij_ddi','db_ij':'n_ij_ddi'}, inplace=True)
# Drop dbij
dfCoadmin.drop(['db_ij'], axis=1, inplace=True)

dfUsers['len_ij'] = df_coadmin['len_ij']
dfUsers['len_ij'] = dfUsers['len_ij'].fillna(0).astype(int)
dfUsers['n_ij'] = df_coadmin['n_ij']
dfUsers['n_ij'] = dfUsers['n_ij'].fillna(0).astype(int)
dfUsers['len_ij_ddi'] = df_inter['len_ij_ddi']
dfUsers['len_ij_ddi'] = dfUsers['len_ij_ddi'].fillna(0).astype(int)
dfUsers['n_ij_ddi'] = df_inter['n_ij_ddi']
dfUsers['n_ij_ddi'] = dfUsers['n_ij_ddi'].fillna(0).astype(int)

# Interactions (subset)
dfInteractions = dfCoadmin.loc[ dfCoadmin['inter']==1 , : ]
# The 'len_ij' in the Interaction DataFrame is called 'len_ij_ddi'
dfInteractions = dfInteractions.rename(columns={'len_ij':'len_ij_ddi'})

print dfDrugs.head()
print dfCoadmin.head()
print dfInteractions.head()
print dfUsers.head()
#
# Export Files
#

print '--- Exporting (gz)Files ---'
with gzip.open('results/dd_drugs.csv.gz', 'wb') as zfile:
	zfile.write(dfDrugs.to_csv(encoding='utf-8', index=False))
#
with gzip.open('results/dd_coadministrations.csv.gz', 'wb') as zfile:
	zfile.write(dfCoadmin.to_csv(encoding='utf-8', index=False))
#
with gzip.open('results/dd_interactions.csv.gz', 'wb') as zfile:
	zfile.write(dfInteractions.to_csv(encoding='utf-8', index=False))
#
with gzip.open('results/dd_users.csv.gz', 'wb') as zfile:
	zfile.write(dfUsers.to_csv(encoding='utf-8', index=True))
