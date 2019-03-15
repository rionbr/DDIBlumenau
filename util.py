# coding=utf-8
# Author: Rion B Correia
# Date: Jan 30, 2015
#
# Util File
# Description: Contains several functions to help handling the DDI files
#
from __future__ import division
import os.path
import pandas as pd
import numpy as np

#
# Given an age, map it to a age group as defined by IBGE
#
ints = np.arange(0, 101, 5)
strs = ['%02d-%02d' % (x, x1-1) for x, x1 in zip(ints[:-1], ints[1:]) ] + ['>99']
def apply_age_group(x):
	g = np.digitize([x], ints, right=False)[0]
	return strs[g-1]

#
# Rename Bairros Names
#
def RenameNeighborhoodNames(df):
	print '- Renaming Neighborhoods'
	# In Regex: (?:) is a non capturing parenthesis
	df['bairro'] = df['bairro'].str.strip()
	#df['bairro'] = df['bairro'].str.upper()

	# Água Verde
	df.loc[ (df['bairro'].isin([u'--GUA VERDE',u'-GUA VERDE',u'AGUA VERDDE',u'AQUA VERDE',u'AGUAR VERDE',u'AGUA VEDEA'])) , ('bairro') ] = 'AGUA VERDE'
	
	# Badenfurt
	df.loc[ (df['bairro'].isin([u'BADENFUT',u'BADENFUR',u'BANDENFURT',u'BADENFURDT',u'BADE',u'BADENFURDET',u'BADENDURT'])), ('bairro') ] = 'BADENFURT'
	# Boa Vista
	# Bom Retiro
	# Centro
	df.loc[ (df['bairro'].isin([u'PETROPOLIS',u'CENTO'])), ('bairro') ] = 'CENTRO' # PETROPOLIS = Incorporado em 2006
	# Escola Agrícola
	df.loc[ (df['bairro'].isin([u'ESVOLA AGRICOLA',u'ESCOLA AGR-COLA',u'ESCOLA AGR--COLA',u'ESCOLA AGRI-COLA',u'ESCOAL AGRICOLA',u'ESCOL AGRICOLA',u'ESC AGRICOLA',u'ESC. AGRICOLA',u'ESC.AGRICOLA',u'ESCOA AGRICOLA',u'ESCOLA AGRICLOLA',u'ASILO',u'AZILO',u'ESCOLA AGRUCOLA'])) , ('bairro') ] = 'ESCOLA AGRICOLA'
	# Fidélis
	df.loc[ (df['bairro'].isin([u'FID-LES',u'FID-LIS',u'FIDELES',u'FIDELLIS',u'FIDELIZ',u'FID--LIS'])) , ('bairro') ] = 'FIDELIS'
	# Fortaleza Alta
	df.loc[ (df['bairro'].isin([u'FORTALEZA-ALTA',u'FORATELZA ALTA'])) , ('bairro') ] = 'FORTALEZA ALTA'
	# Fortaleza
	df.loc[ (df['bairro'].isin([u'FORTAELZA',u'FORETALEZA',u'FORTALENA',u'FORTALEAZA',u'FORTALEAZ',u'FORTAKLEZA',u'FORTAEZA',u'FOETALEZA',u'FORTALZA'])) , ('bairro') ] = 'FORTALEZA'
	# Garcia
	df.loc[ (df['bairro'].isin([u'CARCIA',u'SOUZA CRUZ',u'GACIA',u'GARCIQA',u'GARCIA A128',u'JORDAO'])) , ('bairro') ] = 'GARCIA'
	# Glória
	df.loc[ (df['bairro'].isin([u'GL-RIA',u'GLORIA',u'GL--RIA'])) , ('bairro') ] = 'DA GLORIA'
	# Itoupava Central
	df.loc[ (df['bairro'].str.contains(u'IT(?:.*){0,6} (?:CENTRA|CENTAL|CEMTRAL|CENTRAL|CEBTRAL|CERNTRAL|ALTA|CARDOSO)', regex=True)) , ('bairro') ] = 'ITOUPAVA CENTRAL'
	df.loc[ (df['bairro'].isin([u'ITOUPAVA ALTA',u'TRES COQUEIROS',u'IT.CENTRAL',u'CLINICA JOVENS LIVRES'])) , ('bairro') ] = 'ITOUPAVA CENTRAL'
	# Itoupava Norte
	df.loc[ (df['bairro'].str.contains('IT(?:.*)? NORTE', regex=True)) , ('bairro') ] = 'ITOUPAVA NORTE'
	df.loc[ (df['bairro'].isin([u'SANTA BÁRBARA',u'SANTA MONICA',u'SANTA RITA'])) , ('bairro') ] = 'ITOUPAVA NORTE'
	# Itoupava Seca
	df.loc[ (df['bairro'].isin([u'IT.SECA'])) , ('bairro') ] = 'ITOUPAVA SECA'
	# Itoupavazinha
	df.loc[ (df['bairro'].isin([u'ITOPAVAZINHA',u'ITUOUPAVAZINHA',u'ITOUPAVASINHA',u'ITOUPAZINHA',u'ITOUPVAZINHA',u'ITOUPAVA ZINHA',u'ITOUPACAZINHA',u'JARDIM GERM-NICO',u'DISTRITO INDUSTRIAL'])) , ('bairro') ] = 'ITOUPAVAZINHA'
	# Jardim Blumenau
	df.loc[ (df['bairro'].isin([u'JARDIM DAS AVENIDAS'])) , ('bairro') ] = 'JARDIM BLUMENAU'
	# Nova Esperança
	df.loc[ (df['bairro'].str.contains(u'(?:VILA|NOVA) ESPERAN(?:.*)A', regex=True)) , ('bairro') ] = 'NOVA ESPERANCA'
	df.loc[ (df['bairro'].isin([u'ESPERANCA',u'ESPERAN-A'])) , ('bairro') ] = 'NOVA ESPERANCA'
	# Passo Manso
	df.loc[ (df['bairro'].isin([u'PASSO MANDO',u'PASSOMANSO',u'PASO MANSO',u'PASSO MABSO',u'PASSO MANSO ABRIGO',u'FORTALEZA ALTA - PASSO MANSO',u'RIBEIRAO BRANCO',u'PASSO MANSIO',u'PASSO MANSSO'])) , ('bairro') ] = 'PASSO MANSO'
	# Ponta Aguda
	df.loc[ (df['bairro'].isin([u'BAIRRO PONTA AGUDA'])) , ('bairro') ] = 'PONTA AGUDA'
	# Progresso
	df.loc[ (df['bairro'].isin([u'PROGRESSSO',u'PROGRESO',u'PROGRASSO',u'PROGRESSO ESF SILVANA WITTE',u'PRPGRESSO',u'PORGRESSO',u'CPROGRESSO',u'PREGRESSO',u'PRROGRESSO',u'LIND--IA',u'PROGRES'])) , ('bairro') ] = 'PROGRESSO'
	# Ribeirão Fresco
	df.loc[ (df['bairro'].isin([u'RIBEIR-O FRESCO',u'RIBEIR--O FRESCO'])) , ('bairro') ] = 'RIBEIRAO FRESCO'
	# Salto
	df.loc[ (df['bairro'].isin([u'SALTO',u'PONTE SALTO',u'PONTE SANLTO',u'PONTE DO SALTO'])) , ('bairro') ] = 'DO SALTO'
	# Salto do Norte
	df.loc[ (df['bairro'].isin([u'SALTO NORTE','SALTO DO NORTRE'])) , ('bairro') ] = 'SALTO DO NORTE'
	# Salto Weissbach
	df.loc[ (df['bairro'].isin([u'SLATO WESIBACH',u'SALTO WAISSBBB',u'SALTO WASPBEL',u'VAIS BACK',u'SALTO WEISPALD',u'SALTO WASBAGHT',u'SALTO WEISBACH',u'SALTO W',u'SALTO WASPACK',u'WEISSBACH',u'SALTO WESIBACH',u'SALTO WISBACG',u'SALTO WASSBACH',u'SALTO WESBACH',u'SANTO WEISSBACH'])) , ('bairro') ] = 'SALTO WEISSBACH'
	# Testo Salto
	df.loc[ (df['bairro'].isin([u'TESTO','TESTO CENTRAL',u'TESTO DO SALTO',u'TESTO ALTO',u'TEST SALTO',u'TESTE SALTO'])) , ('bairro') ] = 'TESTO SALTO'
	# Tribess
	df.loc[ (df['bairro'].isin([u'TRIBES',u'TRIBBESS',u'TRIBESS NOSSO A PARTIR 135'])) , ('bairro') ] = 'TRIBESS'
	# Valparaíso
	df.loc[ (df['bairro'].isin([u'VALP--RAISO',u'VALPARA-SO',u'VALAPARISO',u'VALPARISO',u'VALPARAIZO',u'VALPARA--SO',u'VALP--AISO',u'VALAPARAISO',u'ALTO CREDRO'])) , ('bairro') ] = 'VALPARAISO'
	# Velha
	df.loc[ (df['bairro'].isin([u'VELHAA',u'VELHA PEQUENA',u'VALHA',u'VEHA',u'VERLHA',u'BVELHA',u'VEKLHA',u'VELHA  329 1211 CUNHADA',u'VEL-HA',u'CRISTO REI',u'GERAL TRES BARRAS'])) , ('bairro') ] = 'VELHA'
	# Velha Central
	df.loc[ (df['bairro'].isin([u'VELHA CCENTRAL',u'VELHA GENTRAL',u'VELHA CETRAL',u'SANTA CRUZ',u'RISTOW'])) , ('bairro') ] = 'VELHA CENTRAL'
	# Velha Grande
	df.loc[ (df['bairro'].isin([u'VWELHA GRANDE',u'VELH AGRANDE',u'VELHA GRABNDE'])) , ('bairro') ] = 'VELHA GRANDE'
	# Victor Konder
	df.loc[ (df['bairro'].isin([u'VITOR KONDER'])) , ('bairro') ] = 'VICTOR KONDER'
	# Vila Formosa
	# Vila Itoupava
	df.loc[ (df['bairro'].isin([u'ITOUPAVA REGA',u'VILA ITOPAVA'])) , ('bairro') ] = 'VILA ITOUPAVA'
	# Vila Nova
	df.loc[ (df['bairro'].isin([u'VILA  NOVA',u'VILANOVA',u'VILA VOVA'])) , ('bairro') ] = 'VILA NOVA'
	# Vorstadt
	df.loc[ (df['bairro'].isin([u'WORSTART',u'VORSTADTH',u'VORSTART',u'VOSRTADT',u'VORSTARDT',u'BELA VISTA',u'VORSTRT',u'VOSRATRD',u'VRSTADT',u'VORSDAT',u'VORST',u'VOSTARDT',u'VORDST'])) , ('bairro') ] = 'VORSTADT' # Bela Vista é Gaspar, arredondado para Vorstadt
	
	#NAN
	df.loc[ (df['bairro'].isin([u'',u'AUSENTE',u'ATUALIZAR ENDERECO',u'NAO INFORMADO',u'NAO SEI',u'NA----ES',u'N--O SEI',u'N-O SEI'])) , ('bairro') ] = 'OTHER'

	# Outros
	outros = [
		u'-',
		u'.',
		u'0000',
		u'00000',
		u'000000',
		u'07 DE SETEMBRO',
		u'166',
		u'25 DE JULHO',
		u'34',
		u'4546465',
		u'461',
		u'A',
		u'AA',
		u'AAA',
		u'ABRIGO CASA ELISA',
		u'AGUA BRANCA',
		u'AGUA SANTA',
		u'ALBATROZ',
		u'ALTA FLORESTA',
		u'ALTO PINHAL',
		u'ARAPONGUINHAS',
		u'AREAL',
		u'ARIRIB--',
		u'ARRAIAL',
		u'ASALEIA',
		u'AUTO BENEDITO',
		u'BA',
		u'BAIRRO N SRA DE LOURDES',
		u'BARRA DO RIO CERRO',
		u'BARRAC--O',
		u'BARRACAO',
		u'BATEIAS',
		u'BELCHIOR',
		u'BELCHIOR ALTO',
		u'BELCHIOR CENTRAL',
		u'BELO HORIZONTE',
		u'BENEDITO',
		u'BIGORRILHO',
		u'BLOCO E AP 45',
		u'BLUMANAU',
		u'BLUMENAU',
		u'BOCAINA',
		u'BOM PASTOR',
		u'BOMBAS',
		u'BOQUEIRAO',
		u'BRA-O DO BAU',
		u'BRACINHO',
		u'CABO LUIZ QUEVEDO',
		u'CAJAZEIRAS',
		u'CAPITAIS',
		u'CAPOEIRAS',
		u'CASA',
		u'CASCAVEL',
		u'CC',
		u'COHAB',
		u'COLONINHA',
		u'CONFIRMAR',
		u'CONRADINHO',
		u'CRISTO REI SUCUPIRA',
		u'D',
		u'DD',
		u'DEVE COMPROVANTE 20/05',
		u'DIAMANTE',
		u'DOM JOAQUIM',
		u'DOS CA-ADORES',
		u'DOS ESTADOS',
		u'DOS LAGOS',
		u'ENCANO BAIXO',
		u'ENCANO DO NORTE',
		u'ESPINHEIROS',
		u'ESTRADA DAS AREIAS',
		u'F',
		u'FAZENDA SANTIN',
		u'FF',
		u'FIGUEIRA',
		u'FIGUEIRAS',
		u'FORCA----O',
		u'GASPAR',
		u'GASPAR ALTO',
		u'GASPAR GRANDE',
		u'GASPARINHO',
		u'GIRASSOL',
		u'GRAVAT-',
		u'GRAVATA',
		u'GUARANI',
		u'GUARITUBA',
		u'HH',
		u'HUMAIT-',
		u'I',
		u'IBIRAPUITA',
		u'IMIGRANTES',
		u'INFORMA',
		u'INTERIOR',
		u'ITAUM',
		u'ITO',
		u'ITOP.',
		u'ITOUPAVA',
		u'JACO DALFOFO',
		u'JARDIM AMERICA',
		u'JARDIM DAS PALMEIRAS',
		u'JARDIM IPANEMA',
		u'JARDIM IRIRI--',
		u'JARDIM JAQUELINE',
		u'JARDIM JOSE RUPP',
		u'JARDIM LUIS XV',
		u'JARDIM PARAISO',
		u'JENIPAPO',
		u'JHGJ',
		u'JKJLKLJ',
		u'JO--O ZACCO',
		u'JURERE',
		u'KK',
		u'L',
		u'LEBLON (VENDA NOVA)',
		u'LIMEIRA',
		u'LL',
		u'LLL',
		u'LOT PRIMAVERA',
		u'MARGEM ESQUERDA',
		u'MAXIMO',
		u'MMMM',
		u'MMMMMMMMMM',
		u'MOEMA',
		u'MORA EM APIUNA N-O TEM CEP L-',
		u'MORADA DO SOL',
		u'MUCHA',
		u'NONE',
		u'NOSSA SENHORA DE F-TIMA',
		u'NOVA BRASILIA',
		u'NOVA RODEIO',
		u'NOVA TRENTO',
		u'OFICINAS',
		u'PALMARES',
		u'PARQUE INDEPEDENCIA',
		u'POMERANOS',
		u'PONTE DO IMARUIM',
		u'PORTA DO OESTE I',
		u'POSSO GRANDE',
		u'PRAIA BRAVA',
		u'R',
		u'RIBEIR--O S--O LUIS',
		u'RIBEIR-O DAS ANTAS',
		u'RIO DAS PEDRAS',
		u'RIO DO PEIXE',
		u'RIO MORTO',
		u'RISROW',
		u'RODEIO 12',
		u'RODOVIA',
		u'RUA JOSE SEBT',
		u'RURAL',
		u'S--O MIGUEL',
		u'S--O PEDRO',
		u'S--O VICENTE',
		u'S-O CRISTOV-O',
		u'S-O JORGE II',
		u'S-O JOSE',
		u'S-O JOSE OPERARIO',
		u'SANTA B-RBARA',
		u'SANTA CLARA',
		u'SANTA TEREZINHA',
		u'SANTO ANTONIO',
		u'SAO CRISTOVAO',
		u'SAO FRANCISCO',
		u'SAO GABRIEL',
		u'SAO JOAO',
		u'SAO JUDAS',
		u'SAO LUIZ',
		u'SAO PEDRO',
		u'SAO PETESBURGO',
		u'SAUDE',
		u'SDF',
		u'SEM BAIRRO',
		u'SEMIN--RIO',
		u'SENHORINHA',
		u'SERAFIM',
		u'SETE DE SETEMBRO - GASPAR',
		u'SITIO',
		u'SITIO CERCADO',
		u'SOL',
		u'SS',
		u'SSS',
		u'T',
		u'TAPAJOS',
		u'TAPERA',
		u'TATUQUARA',
		u'TRINTA REIS',
		u'V',
		u'VALERIA HOSTIN',
		u'VARGEM GRANDE',
		u'VILA ANTUNES',
		u'VILA BROMBERG',
		u'VILA GERMER',
		u'WARNOW',
		u'WUNDER WALD',
		u'X',
		u'XAXIM',
		u'XX',
		u'XXXX',
		u'XXXXX',
		u'XXXXXXX',
		u'Z',
		u'ZONA RURAL',
		]

	df.loc[ (df['bairro'].isin(outros)) , ('bairro') ] = 'OTHER'

	return df

"""
def dfBnu():
	df = pd.read_csv('data/dados_blumenau_censo2010.csv', header=0, encoding='utf-8')
	df = df.groupby('Cod_bairro').agg({
		'Cod_bairro':'first',
		'Nome_do_bairro':'first',
		'n_homes':'sum',
		'nr_residentes':'sum',
		'nr_homens':'sum',
		'nr_mulheres':'sum',
		})
	#df.rename(columns={'cod_bairro':'id_hood','n_mulheres':'females','n_homens':'males','n_residentes':'residents'}, inplace=True)
	df.loc[ (4202404000), ('bairro')] = 'OTHER'
	df.set_index('bairro',drop=True,inplace=True)

	df['gender_rate'] = df['nr_mulheres'] / df['nr_homens']
	#df['p(M)'] = df['n_male'] / df['residents']
	df.sort_values('nr_residentes',ascending=True, inplace=True)
	return df
"""

def dfCenso(age_per_gender=False):
	df = pd.read_csv('data/Base_informacoes_setores2010_sinopse_Blu.csv', index_col=0, header=5, encoding='utf-8', na_values='X', nrows=522)
	df.rename(columns={
		'V001':'n_homes',
		'Situacao_setor':'type',
		'V014':'population',
		'V015':'males',
		'V016':'females',
		'Nome_do_bairro':'hood'
	}, inplace=True)
	dict_tipo_setor = {1:'City Urban',2:'City Non-Urban',3:'Isolated',4:'Rural',5:'Rural',6:'Rural',7:'Rural',8:'Rural'}
	df['type'].replace(dict_tipo_setor, inplace=True)
	
	# Adjust for 2016 projection
	pop2010 = 309011
	proj2015 = 338876
	df['population'] = df['population'].map(lambda x : x / pop2010 * proj2015)
	df['males'] = df['males'].map(lambda x : x / pop2010 * proj2015)
	df['females'] = df['females'].map(lambda x : x / pop2010 * proj2015)
	if age_per_gender:
		# Ages per Gender
		df['M 00-04'] = df[['V073','V074','V075','V076','V077']].sum(axis=1)
		df['M 05-09'] = df[['V078','V079','V080','V081','V082']].sum(axis=1)
		df['M 10-14'] = df[['V083','V084','V085','V086','V087']].sum(axis=1)
		df['M 15-19'] = df[['V088','V089','V090','V091','V092']].sum(axis=1)
		df['M 20-24'] = df[['V093','V094','V095','V096','V097']].sum(axis=1)
		ages_M_cols = ['V%03d'%d for d in range(98,114)]
		ages_M_names = ['%s'%d for d in ['M 25-29','M 30-34','M 35-39','M 40-44','M 45-49','M 50-54','M 55-59','M 60-64','M 65-69','M 70-74','M 75-79','M 80-84','M 85-89','M 90-94','M 95-99','M >99']]
		ages_M_rename = {k:v for k,v in zip(ages_M_cols, ages_M_names)}
		df.rename(columns=ages_M_rename, inplace=True)
		df['F 00-04'] = df[['V114','V115','V116','V117','V118']].sum(axis=1)
		df['F 05-09'] = df[['V119','V120','V121','V122','V123']].sum(axis=1)
		df['F 10-14'] = df[['V124','V125','V126','V127','V128']].sum(axis=1)
		df['F 15-19'] = df[['V129','V130','V131','V132','V133']].sum(axis=1)
		df['F 20-24'] = df[['V134','V135','V136','V137','V138']].sum(axis=1)
		ages_F_cols = ['V%03d'%d for d in range(139,155)]
		ages_F_names = ['%s'%d for d in ['F 25-29','F 30-34','F 35-39','F 40-44','F 45-49','F 50-54','F 55-59','F 60-64','F 65-69','F 70-74','F 75-79','F 80-84','F 85-89','F 90-94','F 95-99','F >99']]
		ages_F_rename = {k:v for k,v in zip(ages_F_cols, ages_F_names)}
		df.rename(columns=ages_F_rename, inplace=True)
		all_ages_names = ['M 00-04','M 05-09','M 10-14','M 15-19','M 20-24'] + ages_M_names + ['F 00-04','F 05-09','F 10-14','F 15-19','F 20-24'] + ages_F_names
	else:
		df['00-04'] = df[['V032','V033','V034','V035','V036']].sum(axis=1)
		df['05-09'] = df[['V037','V038','V039','V040','V041']].sum(axis=1)
		df['10-14'] = df[['V042','V043','V044','V045','V046']].sum(axis=1)
		df['15-19'] = df[['V047','V048','V049','V050','V051']].sum(axis=1)
		df['20-24'] = df[['V052','V053','V054','V055','V056']].sum(axis=1)
		ages_cols = ['V%03d'%d for d in range(57,73)]
		ages_names = ['%s'%d for d in ['25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','>99']]
		ages_rename = {k:v for k,v in zip(ages_cols, ages_names)}
		df.rename(columns=ages_rename, inplace=True)
		all_ages_names = ['00-04','05-09','10-14','15-19','20-24'] + ages_names

	df[all_ages_names] = df[all_ages_names].applymap(lambda x : x / pop2010 * proj2015)

	df['hood'] = df['hood'].str.upper()
	df.loc[ (df['cod_bairro']==4202404000), ('hood')] = 'OTHER'
	df = df[['cod_bairro','type','hood','n_homes','population','males','females'] + all_ages_names]
	df.fillna(0, inplace=True)
	#
	return df

def dfRenda():
	if not os.path.isfile('data/income.csv'):
		df = pd.read_csv('data/DomicilioRenda_Blumenau_2010.csv', index_col=0)
		df.rename(columns={'V002':'income'}, inplace=True)
		df = df[['income']]
		df.to_csv('data/income.csv',encoding='utf-8')
	else:
		df = pd.read_csv('data/income.csv', index_col=0, encoding='utf-8')
	return df

def dfEdu():
	if not os.path.isfile('data/education.csv'):
		df = pd.read_csv('data/Educacao_Maio_2015.csv', index_col=0, encoding='utf-8')
		df.rename(columns={'TIPO':'type'}, inplace=True)
		df.rename(columns={'BAIRRO':'hood'}, inplace=True)
		df['count'] = 1
		df['hood'] = df['hood'].str.upper()
		#
		df = df.groupby(['type','hood']).agg({'count':'sum'})
		df = df.unstack(level=1).T
		df.columns = ['n_CEI','n_state_school','n_city_school','n_university']
		df.index = df.index.droplevel(0)
		df['n_education_total'] = df.sum(axis=1)
		df.to_csv('data/education.csv',encoding='utf-8')
	else:
		df = pd.read_csv('data/education.csv', index_col=0, encoding='utf-8')
	return df

def dfSaude():
	if not os.path.isfile('data/health.csv'):
		df = pd.read_csv('data/Saude_Janeiro_2015.csv', encoding='utf-8')
		df['count'] = 1
		df = df[['TIPO','BAIRRO','count']]
		df.columns = ['type','hood','count']
		df['hood'] = df['hood'].str.upper()
		df = df.groupby(['type','hood']).agg({'count':'sum'})
		df = df.unstack(level=1).T
		df.columns = ['n_ambulatory','n_ESF','n_hospital','n_UAS']
		df.index = df.index.droplevel(0)
		df['n_health_total'] = df.sum(axis=1)
		df.to_csv('data/health.csv',encoding='utf-8')
	else:
		df = pd.read_csv('data/health.csv', index_col=0, encoding='utf-8')
	return df

def dfSeguranca():
	df = pd.read_csv('data/ssp-blumenau-2015.csv', index_col=0,	encoding='utf-8')
	return df

def CombineDFs(dfCenso, dfRenda, dfEdu, dfSaude, dfSeguranca, age_per_gender):
	""" Combine DF do Censo, de Renda, Educacao e Saude """
	# Add Renda
	dfCenso['income'] = dfRenda['income']
	#
	if age_per_gender:
		dfCenso = dfCenso.groupby('hood').agg(
				{
					#'cod_bairro':'sum',
					#'type':'first',
					'population':'sum',
					'males':'sum',
					'females':'sum',
					'n_homes':'sum',
					'income':'sum',
					'M 00-04':'sum', 'M 05-09':'sum', 'M 10-14':'sum', 'M 15-19':'sum', 'M 20-24':'sum', 'M 25-29':'sum', 'M 30-34':'sum', 'M 35-39':'sum', 'M 40-44':'sum', 'M 45-49':'sum', 'M 50-54':'sum', 'M 55-59':'sum', 'M 60-64':'sum', 'M 65-69':'sum', 'M 70-74':'sum', 'M 75-79':'sum', 'M 80-84':'sum', 'M 85-89':'sum', 'M 90-94':'sum', 'M 95-99':'sum', 'M >99':'sum',
					'F 00-04':'sum', 'F 05-09':'sum', 'F 10-14':'sum', 'F 15-19':'sum', 'F 20-24':'sum', 'F 25-29':'sum', 'F 30-34':'sum', 'F 35-39':'sum', 'F 40-44':'sum', 'F 45-49':'sum', 'F 50-54':'sum', 'F 55-59':'sum', 'F 60-64':'sum', 'F 65-69':'sum', 'F 70-74':'sum', 'F 75-79':'sum', 'F 80-84':'sum', 'F 85-89':'sum', 'F 90-94':'sum', 'F 95-99':'sum', 'F >99':'sum'
					}
			)
	else:
		dfCenso = dfCenso.groupby('hood').agg(
				{
					#'cod_bairro':'sum',
					#'type':'first',
					'population':'sum',
					'males':'sum',
					'females':'sum',
					'n_homes':'sum',
					'income':'sum',
					'00-04':'sum',
					'05-09':'sum',
					'10-14':'sum',
					'15-19':'sum',
					'20-24':'sum',
					'25-29':'sum',
					'30-34':'sum',
					'35-39':'sum',
					'40-44':'sum',
					'45-49':'sum',
					'50-54':'sum',
					'55-59':'sum',
					'60-64':'sum',
					'65-69':'sum',
					'70-74':'sum',
					'75-79':'sum',
					'80-84':'sum',
					'85-89':'sum',
					'90-94':'sum',
					'95-99':'sum',
					'>99':'sum'
				}
			)
	#dfCenso['prob_domicilio'] = dfCenso['n_homes'] / dfCenso['n_homes'].sum()
	#dfCenso['prob_residente'] = dfCenso['population'] / dfCenso['population'].sum()
	dfCenso['gender_rate'] = dfCenso['females'] / dfCenso['males']
	dfCenso['avg_income'] = (dfCenso['income'] / dfCenso['population']).astype(int)
	dfSeguranca['theft_pc'] = dfSeguranca['furto'] / dfCenso['population']
	dfSeguranca['robbery_p1000'] = dfSeguranca['roubo'] / dfCenso['population'] * 1000
	dfSeguranca['suicide_p1000'] = dfSeguranca['suicidio'] / dfCenso['population'] * 1000
	dfSeguranca['transitcrime_p1000'] = dfSeguranca['crimetransito'] / dfCenso['population'] * 1000
	dfSeguranca['traffic_p1000'] = dfSeguranca['trafico'] / dfCenso['population'] * 1000
	dfSeguranca['rape_p1000'] = dfSeguranca['estupro'] / dfCenso['population'] * 1000
	# Sort
	dfCenso.sort_index(axis=1,ascending=True,inplace=True)
	#dfCenso.sort_values(ascending=True, inplace=True)
	#
	dfCenso = pd.concat([dfCenso, dfEdu, dfSaude, dfSeguranca], axis=1, join='outer')
	# Fill Na
	dfCenso.fillna(0,inplace=True)
	return dfCenso


def BnuData(age_per_gender=False):
	df = CombineDFs(dfCenso(age_per_gender),dfRenda(),dfEdu(),dfSaude(),dfSeguranca(),age_per_gender=age_per_gender)
	return df

def dfBnuClima():
	df = pd.read_csv('data/clima_blumenau.csv', index_col=0, encoding='utf-8')
	return df

def dfUsersInteractionsSummary(
	dict_sexo={1:'Male', 2:'Female'},
	dict_escolaridade={1:'Cant read/write', 2:'Can read/write a note', 3:'Incomplete elementary', 4:'Complete elementary', 5:'Incomplete high school', 6:'Complete high school', 7:'Incomplete college', 8:'Complete college', 9:'Espec./Residency', 10:'Masters', 11:'Doctoral', 99:'Not reported'},
	cat_escolaridade=['Cant read/write', 'Can read/write a note', 'Incomplete elementary', 'Complete elementary', 'Incomplete high school', 'Complete high school', 'Incomplete college', 'Complete college', 'Espec./Residency', 'Masters', 'Doctoral', 'Not reported'],
	dict_estado_civil={0:'Not informed', 1:'Married', 2:'Single', 3:'Widower', 4:'Separated', 5:'Consensual union', 6:'Divorced', 7:'Ignored', 9:'Ignored'},
	cat_estado_civil=['Not informed','Single','Consensual union','Married','Widower','Separated','Divorced','Ignored'],
	apply_age_group=apply_age_group,
	cat_age_groups=['00-04', '05-09', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '>99'],
	loadCoAdmin=False,
	):

	# Load Summary
	#dfs = pd.read_csv('results/dd_summary.csv.gz', index_col=0, header=0, encoding='utf-8',
	#	names=['id_user','qt_drugs','qt_drugs_unique','qt_coadmin','qt_coadmin_unique','qt_inter','qt_inter_unique'],
	#	dtype={'id_user':np.int64,'qt_coadmin':np.int16,'qt_coadmin_unique':np.int16,'qt_inter':np.int8,'qt_inter_unique':np.int8})
	#dfs.rename(columns={'qt_interactions':'qt_inter'}, inplace=True)
	#print dfs.head()

	# Load Users
	dfu = pd.read_csv('results/dd_users.csv.gz', index_col=0, header=0, encoding='utf-8',
		#names=['id_user','age','id_city','dob','hood','education','marital','gender','n_drugs','n_drugs_u','n_inter','n_inter_u'],
		dtype={'id_user':np.int64})
	# Map Categorical Data	
	dfu['education'].replace(dict_escolaridade, inplace=True)
	dfu['education'] = dfu['education'].astype('category')
	dfu['education'].cat.set_categories(cat_escolaridade, ordered=True, inplace=True)
	dfu['gender'].replace(dict_sexo, inplace=True)
	dfu['gender'] = dfu['gender'].astype('category')
	dfu['gender'].cat.set_categories(['Male','Female'], ordered=False, inplace=True)
	dfu['marital'].replace(dict_estado_civil, inplace=True)
	dfu['marital'] = dfu['marital'].astype('category')
	dfu['marital'].cat.set_categories(cat_estado_civil, ordered=True, inplace=True)
	dfu['age_group'] = dfu['age'].apply(apply_age_group)
	dfu['age_group'] = dfu['age_group'].astype('category')
	dfu['age_group'].cat.set_categories(cat_age_groups, ordered=True, inplace=True)
	
	#id_outsiders = dfu.loc[ (dfu['id_municipio']!=4347) , : ].index
	#print dfu.head()

	if loadCoAdmin:
		# Load CoAdministrations
		dfc = pd.read_csv('results/dd_coadministrations.csv.gz', header=0, encoding='utf-8', nrows=None,
			#names=['id','id_user','dbi','dien','dipt','count_i','dbj','djen','djpt','count_j','n_coadmin','length','inter','n_inter','text'],
			dtype={'id_user':np.int64})
		dfc['en_ij'] = dfc['en_i'].str.cat(dfc['en_j'], sep='-')
		dfc['db_ij'] = dfc['db_i'].str.cat(dfc['db_j'], sep='-')
		# Separate the Interactions
		dfi = dfc.loc[ (dfc['inter']==1) , : ].copy()
		# Rename the 'len_ij' to 'len_ij_ddi'
		dfi = dfi.rename(columns={'len_ij':'len_ij_ddi'})

	else:

		dfi = pd.read_csv('results/dd_interactions.csv.gz', header=0, encoding='utf-8', nrows=None,
			#names=['id','id_user','dbi','dien','dipt','count_i','dbj','djen','djpt','count_j','n_coadmin','overlap_length','inter','n_inter','text'],
			dtype={'id_user':np.int64})
		dfi['en_ij'] = dfi.apply(lambda x:'%s-%s' % (x['en_i'],x['en_j']), axis=1)
		dfi['db_ij'] = dfi.apply(lambda x:'%s-%s' % (x['db_i'],x['db_j']), axis=1)
		#dfi['count'] = 1

		# Separate the CoADmin (empty!)
		dfc = dfi.loc[ (dfi['inter']==0) , : ]

	# Add Drugs.com Classification
	dfddc = pd.read_csv('dictionaries/drugs.com2.csv', index_col=0, encoding='utf-8')
	dict_interaction_class = dfddc['severity'].to_dict()
	dfi['severity'] = dfi['en_ij'].map(lambda x : dict_interaction_class[x]).astype('category')
	cat_severity = ['Major','Moderate','Minor','None','*']
	dfi['severity'].cat.set_categories(cat_severity, ordered=True, inplace=True)

	#Add Drugs.com DrugType
	dftp = pd.read_csv('dictionaries/drug_classes.csv', index_col=0, encoding='utf-8')
	dict_drug_type = dftp['class'].to_dict()
	dfi['class_i'] = dfi['db_i'].map(lambda x : dict_drug_type[x])
	dfi['class_j'] = dfi['db_j'].map(lambda x : dict_drug_type[x])
	dfi['class_ij'] = dfi.apply(lambda x:'%s-%s' % (x['class_i'],x['class_j']), axis=1)

	return dfu, dfc, dfi
