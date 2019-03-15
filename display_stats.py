# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot DDI Statistics
#
#
# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
slice = pd.IndexSlice
pd.set_option('display.max_rows', 24)
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 300)
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import util
from collections import OrderedDict
import math


def calc_conf_interval(r, **kwargs):
	df = kwargs['n_runs']-1
	mean = r.iloc[0]
	std = r.iloc[1]
	sigma = std/math.sqrt(n_runs)
	(ci_min,ci_max) = stats.t.interval(alpha=0.95, df=n_runs-1, loc=mean, scale=sigma)
	return pd.Series([ci_min, ci_max], index=['ci_min', 'ci_max'])

#
# Load CSVs
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=False)
#dfig = dfi.groupby('id_user').agg({'n_drugs':'sum','n_ij_ddi':'sum','n_coadmin':'sum'})

dfd = pd.read_csv('results/dd_drugs.csv.gz', header=0, encoding='utf-8',
		names=['id_user','DB_label','count','en_i'],
		dtype={'id_user':np.int64})


print '>> dfu'
print dfu.head()
print dfu.shape
print '>> dfc'
print dfc.head()
print dfc.shape
print '>> dfi'
print dfi.head()
print dfi.shape
print '>> dfd'
print dfd.head()
print dfd.shape


dfiu = pd.merge(dfi, dfu[['gender','age','age_group']], how='left', left_on='id_user', right_index=True)

print '>> dfiu'
print dfiu.head()

print '--- --- ---'

#
# Removed Hormones
#
female_hormones = ['Ethinyl Estradiol','Estradiol','Norethisterone','Levonorgestrel','Estrogens Conj.']
dfiu_nh = dfiu.loc[ (~(dfiu['en_i'].isin(female_hormones)) & ~(dfiu['en_j'].isin(female_hormones)) ) , : ].reset_index(drop=True)
dfu['len_ij_ddi_not_hormone'] = dfiu_nh['len_ij_ddi']
dfiug_nh = dfiu_nh.groupby('id_user').agg({'len_ij_ddi':'sum','id_user':'count'})
dfu['n_ij_ddi_not_hormone'] = dfiug_nh['id_user']
print dfu.loc[ dfu['len_ij_ddi']>dfu['len_ij_ddi_not_hormone'], : ].head()

#
# Variables
#
n_user        = len( dfu.index.unique() )
n_user_adult  = len( dfu.loc[ (dfu['age']>=20), : ].index.unique() )
n_user_male   = len( dfu.loc[ (dfu['gender']=='Male'), : ].index.unique() )
n_user_female = len( dfu.loc[ (dfu['gender']=='Female'), : ].index.unique() )
n_user_40p  = len( dfu.loc[ ( (dfu['age']>=40) ), :].index.unique() )
n_user_65p    = len( dfu.loc[ ( (dfu['age']>=66) ), :].index.unique() )

n_a = dfu['n_a'].sum()
n_i = dfd.groupby('DB_label').agg({'en_i':'first'}).shape[0]
n_i_inter = len(np.unique(dfi[['db_i','db_j']].values))
n_ij = dfu['n_ij'].sum()
n_ij_ddi = dfu['n_ij_ddi'].sum()
n_ij_ddi_unique = len( dfiu['db_ij'].unique() )
n_ij_ddi_unique_nh = len( dfiu_nh['db_ij'].unique() )

n_user_gt2drugs = len( dfu.loc[ (dfu['n_i']>1), : ].index.unique() )

n_user_gt1coadmin = len( dfu.loc[ (dfu['n_ij']>0), : ].index.unique() )
n_user_male_ij = len( dfu.loc[ ((dfu['gender']=='Male')  & (dfu['n_ij']>0)), : ].index.unique() )
n_user_female_ij = len( dfu.loc[ ((dfu['gender']=='Female') & (dfu['n_ij']>0)), : ].index.unique() )

n_user_ij_ddi = len( dfu.loc[ (dfu['n_ij_ddi']>0), : ].index.unique() )
n_user_ij_ddi_major = len( dfiu.loc[ (dfiu['severity'].isin(['Major'])) , : ]['id_user'].unique()  )
n_user_adult_ij_ddi = len( dfu.loc[ ((dfu['age']>=20) & (dfu['n_ij_ddi']>0)), : ].index.unique() ) 
n_user_adult_ij_ddi_major = len( dfiu.loc[ ((dfiu['age']>=20) & (dfiu['severity'].isin(['Major']))) , : ]['id_user'].unique()  )

n_males_qt1inter = len( dfu.loc[ ((dfu['gender']=='Male')  & (dfu['n_ij_ddi']>0)), : ].index.unique() )
n_females_qt1inter = len( dfu.loc[ ((dfu['gender']=='Female') & (dfu['n_ij_ddi']>0)), : ].index.unique() )

n_males_qt1inter_nh = len( dfu.loc[ ((dfu['gender']=='Male') & (dfu['n_ij_ddi_not_hormone']>0)), : ].index.unique() )
n_females_qt1inter_nh = len( dfu.loc[ ((dfu['gender']=='Female') & (dfu['n_ij_ddi_not_hormone']>0)), : ].index.unique() )

n_user_ij_ddi_40p = len( dfu.loc[ ((dfu['n_ij_ddi']>0) & (dfu['age']>=40) ), :].index.unique() )
n_user_ij_ddi_major_65p = len( dfiu.loc[ ((dfiu['age']>=66) & (dfiu['severity'].isin(['Major']))) , : ]['id_user'].unique()  )
##

print '--- RRC/RRI direct computation ---'
p_female = n_user_female/n_user
p_male = n_user_male/n_user
print 'P(u^[F]) = {:,.4f}'.format( (p_female) )
print 'P(u^[M]) = {:,.4f}'.format( (p_male) )
print
RRCF = (n_user_female_ij/n_user_female)/(n_user_male_ij/n_user_male)
RRIF = (n_females_qt1inter/n_user_female)/(n_males_qt1inter/n_user_male)
print 'RRC^[F] = ( |U^[c,F]| / |U^[F]| ) / ( |U^[c,M]| / |U^[M]| )  = ({:,d} / {:,d}) / ({:,d}/{:,d}) = {:,.4f}'.format( n_user_female_ij,n_user_female,n_user_male_ij,n_user_male,RRCF  )
print 'RRI^[F] = ( |U^[c,F]| / |U^[F]| ) / ( |U^[c,M]| / |U^[M]| )  = ({:,d} / {:,d}) / ({:,d}/{:,d}) = {:,.4f}'.format( n_females_qt1inter,n_user_female,n_males_qt1inter,n_user_male,RRIF )


#print 'P(u^{i*}) = No Hormones'
#p_iNHf = n_females_qt1inter_nh/n_user
#p_iNHm = n_males_qt1inter_nh/n_user
#print 'P(I*>0|g=F) / P(I*>0|g=F) = {:,.4f} / {:,.4f} = {:,.4f}'.format( (p_iNHf/p_f) , (p_iNHm/p_m) , (p_iNHf/p_f)/(p_iNHm/p_m) )


# Load BNU
dfBnu = util.BnuData(age_per_gender=False)
city_pop = int(dfBnu['population'].sum())
city_pop_males = int(dfBnu['males'].sum())
city_pop_females = int(dfBnu['females'].sum())

city_pop_adults = int(dfBnu.iloc[:,4:21].sum().sum())

# Load Censo
#dfCenso = util.dfCenso(age_per_gender=False)

#
# Overall Statistics
#
print '--- Overall Statistics ---'
print "Blumenau population: {:,d}".format(city_pop)
print "Blumenau Males: {:,d}".format(city_pop_males)
print "Blumenau Females: {:,d}".format(city_pop_females)
print

print "Pronto population: {:,d} ({:.2%} of Blumenau)".format(n_user, n_user/city_pop)
print "Pronto males: {:,d} ({:.2%})".format(n_user_male, n_user_male/n_user)
print "Pronto females: {:,d} ({:.2%})".format(n_user_female, n_user_female/n_user)
print

print "Pronto adults (>=20) {:,d}".format(n_user_adult)
print "Unique drugs: {:,d}".format(n_i)
print "Unique drugs involved in DDI: {:,d}".format(n_i_inter)
print "Drugs intervals dispensed: {:,d}".format(n_a)
print "Co-administrations: {:,d}".format(n_ij)
print "Interactions: {:,d} ({:.2%})".format(n_ij_ddi, n_ij_ddi/n_ij )
print "Unique DDI pairs: {:,d}".format(n_ij_ddi_unique)
print "Unique DDI pairs (not hormones): {:,d}".format(n_ij_ddi_unique_nh)
print "Patients with 2+ drugs dispensed: {:,d} ({:.2%})".format(n_user_gt2drugs, n_user_gt2drugs/n_user)
print

print "Patients with 1+ co-administration: {:,d} ({:.2%})".format(n_user_gt1coadmin, n_user_gt1coadmin/n_user)
print "Male patients with 1+ co-administration: {:,d}, ({:.2%})".format(n_user_male_ij , n_user_male_ij/n_user_gt1coadmin)
print "Female patients with 1+ co-administration: {:,d}, ({:.2%})".format(n_user_female_ij , n_user_female_ij/n_user_gt1coadmin)
print 

print "Patients with 1+ DDI: {:,d} ({:.2%} Pronto, {:.2%} Bnu)".format(n_user_ij_ddi, n_user_ij_ddi/n_user, n_user_ij_ddi/city_pop)
print "Male patients with 1+ DDI: {:,d} ({:.2%})".format(n_males_qt1inter, n_males_qt1inter/n_user_ij_ddi)
print "Female patients with 1+ DDI: {:,d} ({:.2%})".format(n_females_qt1inter, n_females_qt1inter/n_user_ij_ddi)
print

print "Adults patients (20+) with 1+ DDI: {:,d} ({:.2%} Pronto Adults, {:.2%} Bnu Adults/{:.2%} Pronto, {:.2%} Bnu)".format(n_user_adult_ij_ddi, n_user_adult_ij_ddi/n_user_adult, n_user_adult_ij_ddi/city_pop_adults, n_user_adult_ij_ddi/n_user, n_user_adult_ij_ddi/city_pop)
print "Adult patients (20+) with 1+ MAJOR DDI: {:,d} ({:.2%} Pronto Adults, {:.2%} Bnu Adults/{:.2%} Pronto, {:.2%} Bnu)".format(n_user_adult_ij_ddi_major, n_user_adult_ij_ddi_major/n_user_adult, n_user_adult_ij_ddi_major/city_pop_adults, n_user_adult_ij_ddi_major/n_user, n_user_adult_ij_ddi_major/city_pop)
print "Elderly patients (40+) with 1+ DDI: {:,d} ({:.2%} of patients with DDI, {:.2%} of 40+ patients)".format(n_user_ij_ddi_40p, n_user_ij_ddi_40p/n_user_ij_ddi, n_user_ij_ddi_40p/n_user_40p)
print "Elderly patients (65+) with 1+ MAJOR DDI: {:,d} ({:.2%} of 65+ patients)".format(n_user_ij_ddi_major_65p, n_user_ij_ddi_major_65p/n_user_65p)



#
# Education Stats (everyone)
#

print 'Education (everyone)'
dfEdu = dfu['education'].value_counts().to_frame()
dfEdu.sort_index(inplace=True)
dfEdu['prob1'] = dfEdu['education'] / dfEdu['education'].sum()
dfEdu['cumsum1'] = dfEdu['prob1'].cumsum()
#dfEdu = dfEdu.iloc[:-1,:]
dfEdu['prob2'] = dfEdu['education'] / dfEdu.iloc[:-1,0].sum()
dfEdu['cumsum2'] = dfEdu['prob2'].cumsum()
print dfEdu.to_latex(escape=False)
print dfEdu.sum()
#
# Education stats (above 25 y-old) 
#

print 'Education (>25 yld)'
dfEdu = dfu.loc[ dfu['age']>=25 , 'education'].value_counts().to_frame()
dfEdu.sort_index(inplace=True)
dfEdu['prob1'] = dfEdu['education'] / dfEdu['education'].sum()
dfEdu['cumsum1'] = dfEdu['prob1'].cumsum()
#dfEdu = dfEdu.iloc[:-1,:]
dfEdu['prob2'] = dfEdu['education'] / dfEdu.iloc[:-1,0].sum()
dfEdu['cumsum2'] = dfEdu['prob2'].cumsum()
print dfEdu.to_latex(escape=False)
print dfEdu.sum()

#
# Age (Just Patient age distribution, nothing more)
#
print 'Age (distribution)'
dfAge = dfu['age_group'].value_counts().to_frame().sort_index()
dfAge['prob'] = dfAge['age_group'] / dfAge['age_group'].sum()
dfAge['cumsum'] = dfAge['prob'].cumsum()
print dfAge
print dfAge.sum()

#
# DDI per Severity
#
print '--- DDI per Severity ---'
dfi_s = dfi.groupby('severity').agg({'inter':'count','id_user': pd.Series.nunique})
dfi_s.rename(columns={'inter':'n_ij_ddi','id_user':'users'}, inplace=True) # RENAME id_usuario to users
dfi_s['i_per'] = dfi_s['n_ij_ddi'] / dfi_s['n_ij_ddi'].sum() * 100
dfi_s['u_per-pronto'] = dfi_s['users'] / n_user  * 100
dfi_s['u_per-pop'] = dfi_s['users'] / city_pop * 100
#dfi_s = dfi_s.rename(index={'NONE':'None'})
columns = ['n_ij_ddi','i_per','users','u_per-pronto','u_per-pop']
print dfi_s.to_latex(columns=columns)
print dfi_s.sum(axis=0)
## Print summing None and *
dfi_s_ = dfi_s
dfi_s_['severity_s'] = pd.Categorical(['Major','Moderate','Minor','None','None'], ordered=True)
dfi_s_ = dfi_s_.groupby('severity_s').agg(sum)
print dfi_s_.to_latex(columns=columns)
## Print only for adult population
dfiu_s = dfiu.loc[ (dfiu['age']>=20), : ].groupby('severity').agg({'id_user':pd.Series.nunique})
dfiu_s.rename(columns={'id_user':'users'}, inplace=True) # RENAME id_usuario to users
dfiu_s['u_per-pronto-adult'] = dfiu_s['users'] / n_user_adult  * 100
print dfiu_s.to_latex()
#print dfiu_s.head()
dfiu_s_ = dfiu_s
dfiu_s_['severity_s'] = pd.Categorical(['Major','Moderate','Minor','None','None'], ordered=True)
dfiu_s_ = dfiu_s_.groupby('severity_s').agg(sum)
print dfiu_s_.to_latex()

## Print summing Major-Moderate and Moderate-Minor
dfi_ = dfi[['severity','inter','id_user']].copy()
dfi_['severity'] = dfi_['severity'].cat.add_categories(['MajorModerate','ModerateMinor'])
dfi_majmod = dfi_.copy()
dfi_modmin = dfi_.copy()
dfi_majmod.loc[ (dfi_majmod['severity'].isin(['Major','Moderate'])) , 'severity'] = 'MajorModerate'
dfi_modmin.loc[ (dfi_modmin['severity'].isin(['Moderate','Minor'])) , 'severity'] = 'ModerateMinor'
dfi_majmod_s = dfi_majmod.groupby('severity').agg({'inter':'count','id_user':pd.Series.nunique})
dfi_modmin_s = dfi_modmin.groupby('severity').agg({'inter':'count','id_user':pd.Series.nunique})
dfi_majmod_s.rename(columns={'inter':'n_ij_ddi','id_user':'users'}, inplace=True) # RENAME id_usuario to users
dfi_modmin_s.rename(columns={'inter':'n_ij_ddi','id_user':'users'}, inplace=True) # RENAME id_usuario to users
dfi_majmod_s['i_per'] = dfi_majmod_s['n_ij_ddi'] / dfi_majmod_s['n_ij_ddi'].sum()
dfi_majmod_s['u_per-pronto'] = dfi_majmod_s['users'] / n_user * 100
dfi_majmod_s['u_per-pop'] = dfi_majmod_s['users'] / city_pop * 100
dfi_modmin_s['i_per'] = dfi_modmin_s['n_ij_ddi'] / dfi_modmin_s['n_ij_ddi'].sum()
dfi_modmin_s['u_per-pronto'] = dfi_modmin_s['users'] / n_user * 100
dfi_modmin_s['u_per-pop'] = dfi_modmin_s['users'] / city_pop * 100
print dfi_majmod_s.to_latex(columns=columns)
print dfi_modmin_s.to_latex(columns=columns)

## Print summing Major-Moderate and Moderate-Minor only for ADULTs
dfi_ = dfiu.loc[ (dfiu['age']>=20) , ['severity','id_user']].copy()
dfi_['severity'] = dfi_['severity'].cat.add_categories(['MajorModerate','ModerateMinor'])
dfi_majmod = dfi_.copy()
dfi_modmin = dfi_.copy()
dfi_majmod.loc[ (dfi_majmod['severity'].isin(['Major','Moderate'])) , 'severity'] = 'MajorModerate'
dfi_modmin.loc[ (dfi_modmin['severity'].isin(['Moderate','Minor'])) , 'severity'] = 'ModerateMinor'
dfi_majmod_s = dfi_majmod.groupby('severity').agg({'id_user':pd.Series.nunique})
dfi_modmin_s = dfi_modmin.groupby('severity').agg({'id_user':pd.Series.nunique})
dfi_majmod_s.rename(columns={'id_user':'users'}, inplace=True)
dfi_modmin_s.rename(columns={'id_user':'users'}, inplace=True)
dfi_majmod_s['u_per-pronto-adult'] = dfi_majmod_s['users'] / n_user_adult * 100
dfi_modmin_s['u_per-pronto-adult'] = dfi_modmin_s['users'] / n_user_adult * 100
print dfi_majmod_s.to_latex()
print dfi_modmin_s.to_latex()


#
# DDI per Gender
#
print '--- DDI per Gender ---'
dfi_g = dfiu.groupby('gender').agg({'inter':'count','id_user': pd.Series.nunique})
dfi_g.rename(columns={'inter':'n_ij_ddi','id_user':'users'}, inplace=True) # RENAME id_usuario to users
dfi_g['i_per'] = dfi_g['n_ij_ddi'] / dfi_g['n_ij_ddi'].sum()
dfi_g['u_per-pronto'] = dfi_g['users'] / n_user * 100
dfi_g['u_per-pop'] = dfi_g['users'] / city_pop * 100
columns = ['n_ij_ddi','i_per','users','u_per-pronto','u_per-pop']
print dfi_g.to_latex(columns=columns)
print dfi_g.sum(axis=0)

#
# DDI per Age
#
print '--- DDI per Age ---'
dfu_y = dfu.loc[ (dfu['n_ij_ddi']>0) , : ].reset_index().groupby('age_group').agg({'n_ij_ddi':'sum','id_user':pd.Series.nunique})
dfu_y.rename(columns={'id_user':'n_u_ddi'}, inplace=True) # RENAME id_usuario to users

dfu_y['i_per'] = dfu_y['n_ij_ddi'] / dfu_y['n_ij_ddi'].sum()
dfu_y['u_per-pronto'] = dfu_y['n_u_ddi'] / n_user * 100
dfu_y['u_per-city'] = dfu_y['n_u_ddi'] / city_pop * 100
print dfu_y[['n_ij_ddi','i_per','n_u_ddi','u_per-pronto','u_per-city']].to_latex(escape=False)
print dfu_y.sum(axis=0)
# Print summing None and *
dfu_y = dfu_y.rename(index={'*':'NONE'}).rename(index={'NONE':'n/a'})
print dfu_y.groupby(dfu_y.index).agg(sum)[['n_ij_ddi','i_per','n_u_ddi','u_per-pronto','u_per-city']].to_latex(escape=False)


#
# RRC/RRI per Gender
#
pd.set_option('display.precision', 4)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print '--- RRC/RRI per Gender ---'
dfR_g = pd.concat([
	dfu.reset_index().groupby('gender').agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u'}),
	dfu.loc[ (dfu['n_i']>=2) , :].reset_index().groupby('gender', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{n2}'}),
	dfu.loc[ (dfu['n_ij']>0) , :].reset_index().groupby('gender', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{c}'}),
	dfu.loc[ (dfu['n_ij_ddi']>0) , :].reset_index().groupby('gender', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{i}'})
	], axis=1)
dfR_g.to_csv('csv/gender.csv', encoding='utf-8')
dfR_g['RRC^{F}'] = (dfR_g['u^{c}'] / dfR_g['u']) / (dfR_g.loc['Male','u^{c}'] / dfR_g.loc['Male','u'])
dfR_g['RRI^{F}'] = (dfR_g['u^{i}'] / dfR_g['u']) / (dfR_g.loc['Male','u^{i}'] / dfR_g.loc['Male','u'])
print dfR_g.to_latex(escape=False)


#
# RRC/RRI per Severity and Gender
#
print '--- RRC/RRI per Severity & Gender ---'
dfR_gs = dfiu.groupby(['gender','severity']).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u'})
dfR_gs = dfR_gs.unstack(level=0)
dfR_gs.columns = ['%s^{i,%s}_{s}' % (i,j[0]) for i,j in dfR_gs.columns.values]
dfR_gs['RRI^{F}_{s}'] = (dfR_gs['u^{i,F}_{s}'] / n_user_female) / ( dfR_gs['u^{i,M}_{s}'] / n_user_male)
print dfR_gs.to_latex(escape=False)

#
# RRC/RRI per Age
#
print '--- RRC/RRI per Age ---'
dfR_y = pd.concat([
	dfu.reset_index().groupby('age_group').agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u'}),
	dfu.loc[ (dfu['n_i']>=2) , :].reset_index().groupby('age_group', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{n2}'}),
	dfu.loc[ (dfu['n_ij']>0) , :].reset_index().groupby('age_group', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{c}'}),
	dfu.loc[ (dfu['n_ij_ddi']>0) , :].reset_index().groupby('age_group', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{i}'})
	], axis=1)

# Make Short Table (Concatenating edges values)
df00_89 = dfR_y.iloc[ 0:18, 0:4 ]
df00_89.index = df00_89.index.add_categories(['90+'])
df90_pl = dfR_y.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
dfRs_y = pd.concat([df00_89, df90_pl], axis=0)

dfR_y.to_csv('csv/age.csv', encoding='utf-8')
dfRs_y['RC^{y}'] = dfRs_y['u^{c}'] / dfRs_y['u^{n2}']
dfRs_y['RI^{y}'] = dfRs_y['u^{i}'] / dfRs_y['u^{c}']
print dfRs_y.to_latex(escape=False)

#dfRs_y.to_csv('csv/age_short.csv', encoding='utf-8')


#
# RRC/RRI per Age and Gender
#
print '--- RRC/RRI per Age and Gender ---'
dfR_gy_u = dfu.reset_index().groupby(['gender','age_group'], sort=False).agg({'id_user':pd.Series.nunique}).astype(np.int64)
dfR_gy_u.rename(columns={'id_user':'u'}, inplace=True)
dfR_gy_n = dfu.loc[ (dfu['n_i']>=2) , :].reset_index().groupby(['gender','age_group'], sort=False).agg({'id_user':pd.Series.nunique}).astype(np.int64)
dfR_gy_n.rename(columns={'id_user':'u^{n2}'}, inplace=True)
dfR_gy_c = dfu.loc[ (dfu['n_ij']>0) , :].reset_index().groupby(['gender','age_group'], sort=False).agg({'id_user':pd.Series.nunique}).astype(np.int64)
dfR_gy_c.rename(columns={'id_user':'u^{c}'}, inplace=True)
dfR_gy_i = dfu.loc[ (dfu['n_ij_ddi']>=1) , :].reset_index().groupby(['gender','age_group'], sort=False).agg({'id_user':pd.Series.nunique}).astype(np.int64)
dfR_gy_i.rename(columns={'id_user':'u^{i}'}, inplace=True)

for (gender,dftmp_u), (_,dftmp_n), (_, dftmp_c), (_,dftmp_i) in zip(dfR_gy_u.groupby(level=0), dfR_gy_n.groupby(level=0), dfR_gy_c.groupby(level=0), dfR_gy_i.groupby(level=0)):
	print gender
	dfR_gy = pd.concat([dftmp_u,dftmp_n,dftmp_c,dftmp_i], axis=1)
	dfR_gy.index = dfR_gy.index.droplevel(level=0)

	# Make Short Table (Concatenating edges values)
	df00_89 = dfR_gy.iloc[ 0:18, 0:4 ]
	df90_pl = dfR_gy.iloc[ 18: , 0:4 ].sum(axis=0).to_frame(name='90+').T
	dfRs_gy = pd.concat([df00_89, df90_pl], axis=0)
	
	dfR_gy.to_csv('csv/age_%s.csv' % (gender.lower()), encoding='utf-8')
	dfRs_gy['RC^{y}'] = dfRs_gy['u^{c}'] / dfRs_gy['u^{n2}']
	dfRs_gy['RI^{y}'] = dfRs_gy['u^{i}'] / dfRs_gy['u^{c}']
	print dfRs_gy.to_latex(escape=False)

# Statistical Test Males and Females distribution per age are different
ui_gy_m = dfR_gy_i.loc[ slice['Male',:] , 'u^{i}'].values
ui_gy_f = dfR_gy_i.loc[ slice['Female',:] , 'u^{i}'].values
tstat, pvalue = stats.chisquare(ui_gy_f, f_exp=ui_gy_m)
print 'Chi Square the two samples are independent'
print 't-stat: {:.4f}, p-value: {:.4f}'.format(tstat, pvalue)
KS, pvalue = stats.ks_2samp(ui_gy_m, ui_gy_f)
print 'Kolmogorov-Sminov statistic two samples came from the same continuos distribution'
print 't-stat: {:.4f}, p-value: {:.4f}'.format(KS, pvalue)

#
# RRC/RRI per Number of Unique Drugs
#
dfR = pd.concat([
	dfu.reset_index().groupby('n_i', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u'}),
	dfu.loc[ (dfu['n_i']>=2) , :].reset_index().groupby('n_i', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{n2}'}),
	dfu.loc[ (dfu['n_ij']>0) , :].reset_index().groupby('n_i', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{c}'}),
	dfu.loc[ (dfu['n_ij_ddi']>=1) , :].reset_index().groupby('n_i', sort=False).agg({'id_user':pd.Series.nunique}).rename(columns={'id_user':'u^{i}'})
	], axis=1).fillna(0).astype(np.int64)

# Make Short Table (Concatenating edges values)
df00_20 = dfR.iloc[ 0:20, 0:4 ]
df20_pl = dfR.iloc[ 20: , 0:4 ].sum(axis=0).to_frame(name='>20').T
dfRs = pd.concat([df00_20, df20_pl], axis=0)

dfRs['RRC^{y}'] = (dfRs['u^{c}'] / dfRs['u']) / (dfRs.loc[2,'u^{c}'] / dfRs.loc[2,'u'])
dfRs['RRI^{y}'] = (dfRs['u^{i}'] / dfRs['u']) / (dfRs.loc[2,'u^{i}'] / dfRs.loc[2,'u'])

# Don't forget to remove the extra zeros manually, Rion
print dfRs.to_latex(escape=False)


#
# RRI^g > x
#
dfR = pd.concat([
	dfiu.loc[ (dfiu['gender']=='Male') , : ].groupby(['db_ij']).agg({'id_user':set}).rename(columns={'id_user':'set(u^{M})'}),
	dfiu.loc[ (dfiu['gender']=='Female') , : ].groupby(['db_ij']).agg({'id_user':set}).rename(columns={'id_user':'set(u^{F})'}),
	dfiu.groupby(['db_ij']).agg({'severity':'first','db_i':'first','db_j':'first'})
	], axis=1, sort=False)
dfR[['set(u^{F})','set(u^{M})']] = dfR[['set(u^{F})','set(u^{M})']].applymap(lambda x:set([]) if isinstance(x,float) else x)
dfR['u^{i,M}'] = dfR['set(u^{M})'].apply(len)
dfR['u^{i,F}'] = dfR['set(u^{F})'].apply(len)
dfR['u^{i}'] = dfR.apply(lambda r:len(r['set(u^{M})'].union(r['set(u^{F})'])), axis=1)

dfR['RRI^{F}'] = (dfR['u^{i,F}'] / n_user_female) / (dfR['u^{i,M}'] / n_user_female)
dfR['RRI^{M}'] = (dfR['u^{i,M}'] / n_user_male) / (dfR['u^{i,F}'] / n_user_female)
print dfR.head()
us,ds = [],[]
for RRIgx in [1,2,3,4,5,6,7,8,9,10]:
	n_i_female = len( np.unique( dfR.loc[ (dfR['RRI^{F}']>=RRIgx) , ['db_i','db_j'] ].values ) )
	n_i_male   = len( np.unique( dfR.loc[ (dfR['RRI^{M}']>=RRIgx) , ['db_i','db_j'] ].values ) )
	n_ij_female = dfR.loc[ (dfR['RRI^{F}']>=RRIgx) , : ].shape[0]
	n_ij_male   = dfR.loc[ (dfR['RRI^{M}']>=RRIgx) , : ].shape[0]
	n_ij_major_female = dfR.loc[ ((dfR['severity']=='Major') & (dfR['RRI^{F}']>=RRIgx)), : ].shape[0]
	n_ij_major_male   = dfR.loc[ ((dfR['severity']=='Major') & (dfR['RRI^{M}']>=RRIgx)), : ].shape[0]
	ds.append( (RRIgx,n_i_female,n_i_male,n_ij_female,n_ij_male,n_ij_major_female,n_ij_major_male) )
	n_user_minrri_female = len( set.union( *dfR.loc[ (dfR['RRI^{F}']>=RRIgx) , 'set(u^{F})'].tolist() ) )
	n_user_minrri_male   = len( set.union( *dfR.loc[ (dfR['RRI^{M}']>=RRIgx), 'set(u^{M})'].tolist() ) )
	n_user_minrri_maj_female = len( set.union( *dfR.loc[ ((dfR['severity']=='Major') & (dfR['RRI^{F}']>=RRIgx)), 'set(u^{F})'].tolist() ) )
	n_user_minrri_maj_male   = len( set.union( *dfR.loc[ ((dfR['severity']=='Major') & (dfR['RRI^{M}']>=RRIgx)), 'set(u^{M})'].tolist() ) )

	us.append( (RRIgx,n_user_minrri_female,n_user_minrri_male,n_user_minrri_maj_female,n_user_minrri_maj_male) )

dfRdr = pd.DataFrame(ds, columns=['RRI>x','d^{F}','d^{M}','ij^{F}','ij^{M}','ij^{F}_{maj}','ij^{M}_{maj}']).set_index('RRI>x')
dfRur = pd.DataFrame(us, columns=['RRI>x','u^{F}','u^{M}','u^{F}_{maj}','u^{M}_{maj}']).set_index('RRI>x')
dfRur['u^{F}-per'] = dfRur['u^{F}'] / n_user_female
dfRur['u^{M}-per'] = dfRur['u^{M}'] / n_user_male
dfRur['u^{F}_{maj}-per'] = dfRur['u^{F}_{maj}'] / n_user_female
dfRur['u^{M}_{maj}-per'] = dfRur['u^{M}_{maj}'] / n_user_male
print dfRdr.to_latex(escape=False)
print dfRur.to_latex(escape=False)


#
# Null Models - Age
# RI^{y}
#
pd.set_option('display.precision', 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8')
n_runs = dfN['run'].max()
dfN['u^{i}_{rnd}'] = dfN['u^{i,F}_{rnd}'] + dfN['u^{i,M}_{rnd}']
dfN = dfN.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	('u^{i}_{rnd}',['mean','std']),
]))
dfN.columns = ['-'.join(col).strip() for col in dfN.columns.values]
dfN00_89 = dfN.iloc[ 0:18, : ]
dfN90_pl = dfN.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfN = pd.concat([dfN00_89, dfN90_pl], axis=0)
dfN['RI^{y}_{rnd}'] = dfN['u^{i}_{rnd}-mean'] / dfR_y['u^{c}']
dfN[['u^{i}_{rnd}-ci_min','u^{i}_{rnd}-ci_max']] = dfN[['u^{i}_{rnd}-mean','u^{i}_{rnd}-std']].apply(calc_conf_interval, axis=1, n_runs=n_runs)
dfN['RI^{y}_{rnd}-ci_min'] = dfN['u^{i}_{rnd}-ci_min'] / dfR_y['u^{c}']
dfN['RI^{y}_{rnd}-ci_max'] = dfN['u^{i}_{rnd}-ci_max'] / dfR_y['u^{c}']
print dfN.to_latex(escape=False)


#
# Null Models - Age & Gender
# RI^{y,g}
dfN = pd.read_csv('csv/age_gender_null.csv', index_col=0, encoding='utf-8')
dfM = pd.read_csv('csv/age_male.csv', index_col=0, encoding='utf-8')
dfF = pd.read_csv('csv/age_female.csv', index_col=0, encoding='utf-8')
n_runs = dfN['run'].max()
dfN['u^{i}_{rnd}'] = dfN['u^{i,F}_{rnd}'] + dfN['u^{i,M}_{rnd}']
dfN = dfN.groupby('age_group').agg(OrderedDict([
	('u',['mean']),
	('u^{i}_{rnd}',['mean','std']),
	('u^{i,F}_{rnd}',['mean','std']),
	('u^{i,M}_{rnd}',['mean','std'])
	]))
dfN.columns = ['-'.join(col).strip() for col in dfN.columns.values]
dfN['u'] = dfM['u'] + dfF['u']
dfN['u^{F}'] = dfF['u']
dfN['u^{M}'] = dfM['u']
dfN['u^{c,F}'] = dfF['u^{c}']
dfN['u^{c,M}'] = dfM['u^{c}']
dfN00_89 = dfN.iloc[ 0:18, : ]
dfN90_pl = dfN.iloc[ 18: , : ].sum(axis=0).to_frame(name='90+').T
dfN = pd.concat([dfN00_89, dfN90_pl], axis=0)
dfN['RI^{y,F}_{rnd}'] = dfN['u^{i,F}_{rnd}-mean'] / dfF['u^{c}']
dfN['RI^{y,M}_{rnd}'] = dfN['u^{i,M}_{rnd}-mean'] / dfM['u^{c}']
# Confidence Interval
dfN[['u^{i,F}_{rnd}-ci_min','u^{i,F}_{rnd}-ci_max']] = dfN[['u^{i,F}_{rnd}-mean','u^{i,F}_{rnd}-std']].apply(calc_conf_interval, axis=1, n_runs=n_runs)
dfN[['u^{i,M}_{rnd}-ci_min','u^{i,M}_{rnd}-ci_max']] = dfN[['u^{i,M}_{rnd}-mean','u^{i,M}_{rnd}-std']].apply(calc_conf_interval, axis=1, n_runs=n_runs)
dfN['RI^{y,F}_{rnd}-ci_min'] = dfN['u^{i,F}_{rnd}-ci_min'] / dfN['u^{c,F}']
dfN['RI^{y,F}_{rnd}-ci_max'] = dfN['u^{i,F}_{rnd}-ci_max'] / dfN['u^{c,F}']
dfN['RI^{y,M}_{rnd}-ci_min'] = dfN['u^{i,M}_{rnd}-ci_min'] / dfN['u^{c,M}']
dfN['RI^{y,M}_{rnd}-ci_max'] = dfN['u^{i,M}_{rnd}-ci_max'] / dfN['u^{c,M}']
print dfN.to_latex(escape=False)




