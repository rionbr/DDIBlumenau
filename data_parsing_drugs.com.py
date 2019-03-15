import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)


dt = pd.read_csv('dictionaries/drug_translation.csv', encoding='utf-8')
dict_ptname_id = dt.set_index('drug_pt')['drugbank_label'].to_dict()
print dt.head()
print dict_ptname_id

dc = pd.read_csv('dictionaries/drugs.com.csv', encoding='utf-8')


def map_name_id(x):
	if x in dict_ptname_id:
		return dict_ptname_id[x]
	else:
		return '---'
dc['d1id'] = dc['d1pt'].map(map_name_id)
dc['d2id'] = dc['d2pt'].map(map_name_id)

print dc.head()

dc.to_csv('dictionaries/drugs.com2.csv', encoding='utf-8')