#dd coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot DDI timelines
#
#
from __future__ import division
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('display.max_colwidth', 100)
#pd.set_option('display.float_format', lambda x: ('%.4f' % x).lstrip('0') )
import util
from collections import OrderedDict

#
# Load CSVs
#
dfd = pd.read_csv('results/dd_drugs.csv.gz', header=0, encoding='utf-8', nrows=None)

# Group by db, and count how many users

dfdu_g = dfd.groupby('DB_label')['id_user'].apply(frozenset)

dfu, dfc, dfi = util.dfUsersInteractionsSummary()

print '>> dfu'
print dfu.head()
print dfu.shape
#print '>> dfc'
#print dfc.head()
#print dfc.shape
print '>> dfi'
print dfi.head()
print dfi.shape
#print '>> dfd'
#print dfd.head()
#print dfd.shape

gender_counts = dfu['gender'].value_counts()
nr_males   = gender_counts['Male']
nr_females = gender_counts['Female']
nr_users   = dfu.shape[0]
p_male = nr_males / nr_users
p_female = nr_females / nr_users


# Add Gender to dfi
dfiu = pd.merge(dfi, dfu[['gender']], how='left', left_on='id_user', right_index=True)

#
#
#
dfi['tau'] = dfi['len_ij_ddi'] / (dfi['len_i'] + dfi['len_j'] - dfi['len_ij_ddi']) 

dfig = dfi.groupby(['db_ij']).agg(
		OrderedDict([
			('id_user', pd.Series.nunique),
			('en_ij','first'),
			('db_i','first'),
			('db_j','first'),
			('en_i','first'),
			('en_j','first'),
			('n_i','sum'),
			('n_j','sum'),
			('len_i','sum'),
			('len_j','sum'),
			('len_ij_ddi',['sum','mean','std']),
			('class_i','first'),
			('class_j','first'),
			('severity','first'),
			('text','first'),
			('tau','sum')
		]))
dfig.columns = dfig.columns.droplevel(level=1)
idx = dfig.columns.tolist()
idx[0]  = 'u_ij'
idx[11] = 'mean(len_ij_ddi)'
idx[12] = 'std(len_ij_ddi)'
dfig.columns = pd.Index(idx)

# Map u_i and u_j
dfig['set(u_i)'] = dfig['db_i'].map(dfdu_g)
dfig['set(u_j)'] = dfig['db_j'].map(dfdu_g)

dfig['u_i'] = dfig['set(u_i)'].apply(len)
dfig['u_j'] = dfig['set(u_j)'].apply(len)
#dfig['u_i-u_j'] = dfig.apply(lambda x: len(x['set(u_i)'].union( x['set(u_j)']) ), axis=1 ) # NOT USED ANYMORE

dfig.drop(['set(u_i)','set(u_j)'], axis='columns', inplace=True)

print dfig.head()
# Gamma
dfig['gamma_ij'] = (dfig['u_ij'] / dfig['u_i']) #.round(decimals=3)
dfig['gamma_ji'] = (dfig['u_ij'] / dfig['u_j']) #.round(decimals=3)
# Tau
dfig['tau'] = dfig['tau'] / dfig['u_ij']



# Pivot (DDI users vs Gender)
dfig_g = pd.pivot_table(dfiu, index='db_ij', columns='gender', values='id_user', aggfunc=pd.Series.nunique).fillna(0).rename(columns={'Male':'u_male','Female':'u_female'}).astype(np.int64)
dfig_g.columns = dfig_g.columns.tolist()

# Pivot (DDI instances vs Gender)
dfig_ij = pd.pivot_table(dfiu, index='db_ij', columns='gender', values='len_ij_ddi', aggfunc='sum').fillna(0).rename(columns={'Male':'len_ij_ddi_male','Female':'len_ij_ddi_female'}).astype(np.int64)
dfig_ij.columns = dfig_ij.columns.tolist()

#print dfig_g.head()
#print dfig_ij.head()
#dfig = pd.concat([dfig, dfig_g,dfig_ij], axis=1, join='outer')
dfig = dfig.merge(dfig_g[['u_male','u_female']], how='left', left_index=True, right_index=True)
dfig = dfig.merge(dfig_ij[['len_ij_ddi_male','len_ij_ddi_female']], how='left', left_index=True, right_index=True)

print '>> dfig'
print dfig.head()

dfig.reset_index(inplace=True)

dfig['rank_u'] = dfig['u_ij'].rank(method='min', ascending=False).astype(int)
dfig['rank_tau'] = dfig['tau'].rank(method='min', ascending=False).astype(int)
dfig['rank_gamma_ij'] = dfig['gamma_ij'].rank(method='min', ascending=False).astype(int)
dfig['rank_gamma_ji'] = dfig['gamma_ji'].rank(method='min', ascending=False).astype(int)
dfig['rankproduct_gamma'] = dfig['rank_gamma_ij'] * dfig['rank_gamma_ji'] 
dfig['rank_gamma_rp'] = dfig['rankproduct_gamma'].rank(method='min', ascending=True).astype(int)
dfig['rankproduct_tau'] = dfig['rank_tau'] * dfig['rank_u']
dfig['rank_tau_rp'] = dfig['rankproduct_tau'].rank(method='min', ascending=True).astype(int)


dfig['P(ij)'] = dfig['u_ij'] / nr_users
dfig['P(ij,g=M)'] = dfig['u_male'] / nr_users
dfig['P(ij,g=F)'] = dfig['u_female'] / nr_users
dfig['P(ij|g=M)'] = dfig['P(ij,g=M)'] / p_male
dfig['P(ij|g=F)'] = dfig['P(ij,g=F)'] / p_female

# Relative Risk
dfig['RRI^F'] = dfig['P(ij|g=F)'] / dfig['P(ij|g=M)']
dfig['RRI^M'] = dfig['P(ij|g=M)'] / dfig['P(ij|g=F)']

print dfig.loc[ (dfig['RRI^F']>=1) , 'u_female'].sum()

# Shorten Names (for Latex)
dfig.loc[ dfig['en_i'] == 'Medroxyprogesterone Acetate', 'en_i'] = 'Medroxyproges. Ac.'
dfig.loc[ dfig['en_j'] == 'Medroxyprogesterone Acetate', 'en_j'] = 'Medroxyproges. Ac.'
dfig.loc[ dfig['en_i'] == 'Acetylsalicylic Acid', 'en_i'] = 'ASA'
dfig.loc[ dfig['en_j'] == 'Acetylsalicylic Acid', 'en_j'] = 'ASA'
dfig.loc[ dfig['en_i'] == 'Estrogens Conjugated', 'en_i'] = 'Estrogens Conj.'
dfig.loc[ dfig['en_j'] == 'Estrogens Conjugated', 'en_j'] = 'Estrogens Conj.'
dfig.loc[ dfig['text'] == 'This anti-infectious agent could decrease the effect  of the oral contraceptive', 'text'] = 'Anti-infectious agent could decrease effect of oral contraceptive'
dfig.loc[ dfig['text'] == 'This anti-infectious agent could decrease the effect of the oral contraceptive', 'text'] = 'Anti-infectious agent could decrease effect of oral contraceptive'
dfig.loc[ dfig['text'] == 'The NSAID decreases the diuretic and antihypertensive effects of the loop diuretic', 'text'] = 'NSAID decreases diuretic and antihypertensive effects of loop diuretic'
dfig.loc[ dfig['text'] == 'Increased digoxin levels and decreased effect in presence of spironolactone', 'text'] = 'Increased digoxin levels and decreased effect with spironolactone'
dfig.loc[ dfig['text'] == 'The anticholinergic increases the risk of psychosis and tardive dyskinesia', 'text'] = 'Anticholinergic inc. risk of psychosis and tardive dyskinesia'
dfig.loc[ dfig['text'] == 'Possible extrapyramidal effects and neurotoxicity with this combination', 'text'] = 'Possible extrapyramidal effects and neurotoxicity'

# Identifying Hormone Drugs
female_hormones = ['Ethinyl Estradiol','Estradiol','Norethisterone','Levonorgestrel','Estrogens Conj.']
dfig_nh = dfig.loc[ (~(dfig['en_i'].isin(female_hormones)) & ~(dfig['en_j'].isin(female_hormones)) ) , : ].reset_index(drop=True)

# Separate Between Male/Female
dfig_m = dfig.loc[ dfig['RRI^M']>1 , : ].copy()
dfig_m['rank_u_male'] = dfig_m['u_male'].rank(method='min', ascending=False).astype(int)
dfig_m['rank_rri'] = dfig_m['RRI^M'].rank(method='min', ascending=False).astype(int)
dfig_m['rankproduct_rri'] = dfig_m['rank_rri'] * dfig_m['rank_u_male']
dfig_m['rank_rri_rp'] = dfig_m['rankproduct_rri'].rank(method='min', ascending=True).astype(int)
dfig_m = dfig_m.sort_values('rank_rri_rp',ascending=True)

dfig_f = dfig.loc[ dfig['RRI^F']>1 , : ].copy()
dfig_f['rank_u_female'] = dfig_f['u_female'].rank(method='min', ascending=False).astype(int)
dfig_f['rank_rri'] = dfig_f['RRI^F'].rank(method='min', ascending=False).astype(int)
dfig_f['rankproduct_rri'] = dfig_f['rank_rri'] * dfig_f['rank_u_female']
dfig_f['rank_rri_rp'] = dfig_f['rankproduct_rri'].rank(method='min', ascending=True).astype(int)
dfig_f = dfig_f.sort_values('rank_rri_rp',ascending=True)

#dfig.to_csv('results/drug_interaction_friendly.csv', columns=['ddi','count','users','class','text','d1en','d2en','d1pt','d2pt'], encoding='utf-8')
# Sort
dfig.sort_values('rank_u', inplace=True, ascending=True)

# Export
dfig.to_csv('csv/ddi.csv', encoding='utf-8')

# Round Numbers
dfig['std(len_ij_ddi)'] = dfig['std(len_ij_ddi)'].fillna(0)
dfig = dfig.round({'mean(len_ij_ddi)':0, 'std(len_ij_ddi)':0, 'gamma_ij':2,'gamma_ji':2, 'tau':2, 'RRI^M':2, 'RRI^F':2})
dfig['mean(len_ij_ddi)'] = dfig['mean(len_ij_ddi)'].astype(int)
dfig['std(len_ij_ddi)'] = dfig['std(len_ij_ddi)'].astype(int)

dfig_f['std(len_ij_ddi)'] = dfig_f['std(len_ij_ddi)'].fillna(0)
dfig_f = dfig_f.round({'mean(len_ij_ddi)':0, 'std(len_ij_ddi)':0, 'gamma_ij':2,'gamma_ji':2, 'tau':2, 'RRI^M':2, 'RRI^F':2})
dfig_f['mean(len_ij_ddi)'] = dfig_f['mean(len_ij_ddi)'].astype(int)
dfig_f['std(len_ij_ddi)'] = dfig_f['std(len_ij_ddi)'].astype(int)

dfig_m['std(len_ij_ddi)'] = dfig_m['std(len_ij_ddi)'].fillna(0)
dfig_m = dfig_m.round({'mean(len_ij_ddi)':0, 'std(len_ij_ddi)':0, 'gamma_ij':2,'gamma_ji':2, 'tau':2, 'RRI^M':2, 'RRI^F':2})
dfig_m['mean(len_ij_ddi)'] = dfig_m['mean(len_ij_ddi)'].astype(int)
dfig_m['std(len_ij_ddi)'] = dfig_m['std(len_ij_ddi)'].astype(int)

pd.set_option('display.precision', 2)

# Some DDI Statistics
n_ij_ddi_unique = dfig.shape[0]
n_ij_ddi_unique_nh = dfig_nh.shape[0]

n_ij_ddi_unique_rriF = len( dfig.loc[ dfig['RRI^F']>1 , 'db_ij'].unique() )
n_ij_ddi_unique_rriM = len( dfig.loc[ dfig['RRI^M']>1 , 'db_ij'].unique() )

n_ij_ddi_unique_nh_rriF = len( dfig_nh.loc[ dfig['RRI^F']>1 , 'db_ij'].unique() )
n_ij_ddi_unique_nh_rriM = len( dfig_nh.loc[ dfig['RRI^M']>1 , 'db_ij'].unique() )

print "Number of unique DDI: {:,d}".format( n_ij_ddi_unique )
print "Number of unique DDI (no hormone): {:,d}".format( n_ij_ddi_unique_nh )
print "Number of unique DDI with RRI^F>1: {:,d}".format( n_ij_ddi_unique_rriF )
print "Number of unique DDI (no hormone) with RRI^F>1: {:,d}".format( n_ij_ddi_unique_nh_rriF )
print "Number of unique DDI with RRI^M>1: {:,d}".format( n_ij_ddi_unique_rriM )
print "Number of unique DDI (no hormone) with RRI^M>1: {:,d}".format( n_ij_ddi_unique_nh_rriM )

# RRI extremes
print '-- RRI Extremes --'
lrri = []
print dfig.shape[0]

for minRRI in [1,2,3,4,5,6,7,8,9,10]:
	
	dfigr_f = dfig_f.loc[ (dfig_f['RRI^F']>=minRRI) , : ]
	dfigr_m = dfig_m.loc[ (dfig_m['RRI^M']>=minRRI) , : ]
	dfigr_f_maj = dfigr_f.loc[ dfigr_f['severity']=='Major', :]
	dfigr_m_maj = dfigr_m.loc[ dfigr_m['severity']=='Major', :]

	#n_i_f = len(np.unique(dfigr_f[['db_i','db_j']]))
	#n_i_m = len(np.unique(dfigr_m[['db_i','db_j']]))

	n_u_f = dfigr_f['u_female'].sum()
	n_u_m = dfigr_m['u_male'].sum()
	
	#n_i_f_maj = len(np.unique(dfigr_f_maj[['db_i','db_j']]))
	#n_i_m_maj = len(np.unique(dfigr_m_maj[['db_i','db_j']]))

	n_u_f_maj = dfigr_f_maj['u_female'].sum()
	n_u_m_maj = dfigr_m_maj['u_male'].sum()

	lrri.append( (minRRI, n_u_f, n_u_m, n_u_f_maj, n_u_m_maj) )
dfrri = pd.DataFrame(lrri, columns=['RRI>x','n_u_f','n_u_m','n_u_f_maj', 'n_u_m_maj'])
print dfrri.head()
dfrri['n_u_f-per-pronto'] = dfrri['n_u_f'] / nr_females * 100
dfrri['n_u_m-per-pronto'] = dfrri['n_u_m'] / nr_males * 100
dfrri['n_u_f_maj-per-pronto'] = dfrri['n_u_f_maj'] / nr_females * 100
dfrri['n_u_m_maj-per-pronto'] = dfrri['n_u_m_maj'] / nr_males * 100

print dfrri.to_latex(index=False, escape=False)
#asd

# To LATEX
print '-- Everything --'
dfig.sort_values('rank_u', inplace=True, ascending=True)
print dfig[['rank_u','u_ij','gamma_ij','tau','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity','text']].to_latex(index=False, escape=False)

print '-- Top 20 (ranked by u) --'
dfig.sort_values('rank_u', inplace=True, ascending=True)
print dfig.loc[ : , ['rank_u','u_ij','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:20].to_latex(index=False, escape=False)

print '-- Top 20 (ranked by gamma (and u, rank product)) --'
dfig.sort_values('rank_gamma_rp', inplace=True, ascending=True)
print dfig.loc[ : , ['rank_gamma_rp','gamma_ij','gamma_ji','u_ij','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:20].to_latex(index=False, escape=False)

#print '-- Top 20 (ranked by gamma ji) --'
#dfig.sort_values('rank_gamma_ji', inplace=True, ascending=True)
#print dfig.loc[ : , ['rank_gamma_ji','gamma_ji','u_ij','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI','severity'] ].iloc[:20].to_latex(index=False, escape=False)

print '-- Top 20 (ranked by tau and u (rank product)) --'
dfig.sort_values('rank_tau_rp', inplace=True, ascending=True)
print dfig.loc[ : , ['rank_tau_rp','rank_tau','rank_u','tau','u_ij','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:20].to_latex(index=False, escape=False)

print '-- Top 20 (ranked by tau) --'
dfig.sort_values('rank_tau', inplace=True, ascending=True)
print dfig.loc[ : , ['rank_tau','tau','u_ij','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:20].to_latex(index=False, escape=False)

print '-- Top 20 Major (ranked by u) ---'
dfig.sort_values('rank_u', inplace=True, ascending=True)
print dfig.loc[ (dfig['severity']=='Major') , ['rank_u','u_ij','gamma_ij','tau','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:20, :].to_latex(index=False, escape=False)

print '-- Top Male DDI --'
print dfig_m.loc[ : , ['rank_rri_rp','rank_rri','rank_u_male','u_male','u_female','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^M','severity'] ].iloc[:70,:].to_latex(index=False, escape=False)

print '-- Top Female DDI --'
print dfig_f.loc[ : , ['rank_rri_rp','rank_rri','rank_u_female','u_male','u_female','mean(len_ij_ddi)','std(len_ij_ddi)','en_i','en_j','RRI^F','severity'] ].iloc[:70,:].to_latex(index=False, escape=False)

print '-- Top Major Male DDI --'
print dfig_m.loc[ (dfig_m['severity']=='Major') , ['rank_rri_rp','u_male','en_i','en_j','RRI^M'] ].iloc[:50,:].to_latex(index=False, escape=False)

print '-- Top Major Female DDI --'
print dfig_f.loc[ (dfig_f['severity']=='Major') , ['rank_rri_rp','u_female','en_i','en_j','RRI^F'] ].iloc[:20,:].to_latex(index=False, escape=False)





