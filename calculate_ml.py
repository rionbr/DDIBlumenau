# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Display results of Interaction on Map
#
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
from matplotlib import pyplot as plt
#import itertools

import numpy as np
#import scipy
from scipy import interp
#from scipy.stats import ttest_ind
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import util
#
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC #, SVC
from sklearn.linear_model import LogisticRegression
#from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, roc_curve, precision_recall_curve, auc
from sklearn.dummy import DummyClassifier
import gzip
from collections import OrderedDict


class RoughClassifier(BaseEstimator, ClassifierMixin):
	
	def _find_min_age(self, x, y, age_min=0, age_max=100):
		min_y, max_mcc = None, None
		for i in range(0,90):
			y_ = np.zeros(shape=y.shape, dtype=int)
			y_[(x>i)] = 1
			mcc = matthews_corrcoef(y, y_)
			if (mcc > max_mcc) or (mcc is None):
				min_y = i
				max_mcc = mcc
		return min_y

	def fit(self, X, y):
		# male, female, age
		age_m = X[(X[:,0] == 1),2]
		age_f = X[(X[:,1] == 1),2]
		y_m = y[(X[:,0] == 1)]
		y_f = y[(X[:,1] == 1)]
		#
		self.min_y_M = self._find_min_age(age_m, y_m)
		self.min_y_F = self._find_min_age(age_f, y_f)
		#
		return self

	def predict(self, X):
		y = np.apply_along_axis(self._rough_rule, axis=1, arr=X)
		return y

	def predict_proba(self, X):
		y = self.predict(X)
		indp = np.where(y==1)
		indn = np.where(y==0)
		P = np.zeros((X.shape[0],2), dtype=np.float64)
		P[indp, 1] = 1.0
		P[indn, 0] = 1.0
		return P

	def _rough_rule(self, x):
		M = x[0] 
		F = x[1]
		age = x[2]
		# Women > age
		if ((F==1) and (age >= self.min_y_M)):
			return 1
		# Men > age
		elif ((M==1) and (age >= self.min_y_F)):
			return 1
		# Else
		else: 
			return 0
#
# Load CSVs
#
dfd = pd.read_csv('results/dd_drugs.csv.gz', header=0, encoding='utf-8', nrows=None)
dfdg = dfd.groupby(['id_user','en_i']).agg({'n_disp':'sum'}).unstack().clip(0,1)
dfdg.columns = dfdg.columns.droplevel(level=0)
drug_cols = dfdg.columns.tolist()

dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=False)


print '--- dfd:'
print dfd.shape
print dfd.head()

print '--- dfu:'
print dfu.shape
print dfu.head()

print '--- dfi:'
print dfi.shape
print dfi.head()


# Truncated Education
dfu['education_tnc'] = dfu['education'].replace({
	'Cant read/write'       :'Incomplete high school',
	'Can read/write a note' :'Incomplete high school',
	'Incomplete elementary' :'Incomplete high school',
	'Complete elementary'   :'Incomplete high school',
	'Incomplete high school':'Incomplete high school',
	'Complete high school'  :'Complete high school',
	'Incomplete college'    :'Complete high school',
	'Complete college'      :'Complete high school',
	'Espec./Residency'      :'Complete high school',
	'Masters'               :'Complete high school',
	'Doctoral'              :'Complete high school',
	'Not reported'          :'Not reported'
	})


#Truncated Estado Civil
dfu['marital_tnc'] = dfu['marital'].replace({
	'Not informed'    :'Not informed',
	'Married'         :'Married', 
	'Single'          :'Single', 
	'Widower'         :'Single', 
	'Separated'       :'Single', 
	'Consensual union':'Married',
	'Divorced'        :'Single',
	'Ignored'         :'Ignored', 
	})


# Load BNU
dfBnu = util.BnuData(age_per_gender=False)
print '- dfBnu : Blumenau'
print dfBnu.head()

# Add Neighborhood to every individual
dfu['avg_income']         = dfu['hood'].map(dfBnu['avg_income'])
dfu['theft_pc']           = dfu['hood'].map(dfBnu['theft_pc'])
dfu['robbery_p1000']      = dfu['hood'].map(dfBnu['robbery_p1000'])
dfu['suicide_p1000']      = dfu['hood'].map(dfBnu['suicide_p1000'])
dfu['transitcrime_p1000'] = dfu['hood'].map(dfBnu['transitcrime_p1000'])
dfu['traffic_p1000']      = dfu['hood'].map(dfBnu['traffic_p1000'])
dfu['rape_p1000']         = dfu['hood'].map(dfBnu['rape_p1000'])

# Concat with Drugs Taken
print '- dfud : Merged Data'
dfML = dfu.join(dfdg)
dfML[drug_cols] = dfML[drug_cols].fillna(0)

print '--- dfML:'
print dfML.shape
print dfML.head()

"""
# Normalized n_ij_ddi
#dfML['n_ij_ddi_norm'] = dfML['n_ij_ddi'] / dfML['qt_drugs']

# Remove/Add 'Omeprazole-Clonazepam' - The most common DDI
s_NotOmeprazoleClonazepam = dfi.loc[ ~(dfi['en_ij'].isin(['Omeprazole-Clonazepam'])) , ['id_user'] ].groupby('id_user').agg({'id_user':'count'})
dfu['n_ij_ddi_not_OmeprClona'] = s_NotOmeprazoleClonazepam

# By Class
s_major = dfi.loc[ (dfi['severity']=='Major'), ['id_user']].groupby('id_user').agg({'id_user':'count'})
dfu['n_ij_ddi_major'] = s_major
dfu['n_ij_ddi_major'].fillna(0,inplace=True)

s_moderate = dfi.loc[ (dfi['severity']=='Moderate'), ['id_user']].groupby('id_user').agg({'id_user':'count'})
dfu['n_ij_ddi_moderate'] = s_moderate
dfu['n_ij_ddi_moderate'].fillna(0,inplace=True)

s_minor = dfi.loc[ (dfi['severity']=='Minor'), ['id_user']].groupby('id_user').agg({'id_user':'count'})
dfu['n_ij_ddi_minor'] = s_minor
dfu['n_ij_ddi_minor'].fillna(0,inplace=True)
"""
"""
custom_pairs = [
	#'Digoxin-Hydrochlorothiazide',
	#'Prednisone-Estradiol',
	#'Ethinyl Estradiol-Amoxicillin',
	#'Erythromycin-Simvastatin',
	'Diltiazem-Simvastatin',
	#'Amitriptyline-Fluoxetine',
	#'Imipramine-Fluoxetine',
	#'Fluconazole-Simvastatin'
	]
s_custom = dfi.loc[ (dfi.columns.isin(custom_pairs)) , ['id_user']].groupby('id_user').agg({'id_user':'count'})
dfud['n_ij_ddi_custom'] = s_custom
"""

#
# Machine Learning - Regression / Classification
#
print '- Machine Learning - Classification'

ml_type = 'simple'

if ml_type =='simple':
	cat_cols = ['gender']
	num_cols = ['age','n_i','n_ij']
	drug_cols = dfdg.columns.tolist()
	suffix = 'simple'

elif ml_type == 'complete':
	cat_cols = ['gender','education','marital']
	num_cols = ['age','avg_income','theft_pc','robbery_p1000','suicide_p1000','transitcrime_p1000','traffic_p1000','rape_p1000','n_i','n_ij']
	drug_cols = dfdg.columns.tolist()
	suffix = 'complete'

elif ml_type == 'nodrug':
	cat_cols = ['gender']
	num_cols = ['age','n_i','n_ij']
	drug_cols = []
	suffix = 'nodrug'


cat_features = dfML[ cat_cols ].T.to_dict().values()
num_features = dfML[ num_cols ].values
bin_features = dfML[ drug_cols ]

#
DV = DictVectorizer(sparse=False)
SS = StandardScaler()
#
print '> Vectoring Categorical Features'
X_cat = DV.fit_transform(cat_features)
print '> Scaling Numerical Features'
X_num = SS.fit_transform(num_features)
print '> Binary Features'
X_bin = bin_features

if ml_type == 'simple':
	features = DV.feature_names_ + ['age','n_i','n_ij'] + ['d='+d for d in drug_cols]
elif ml_type == 'complete':
	features = DV.feature_names_ + ['age','avg_income','thefts','robberies','suicides','transitcrimes','trafficking','rapes','qt_drugs','qt_coadmin'] + ['d_'+d for d in drug_cols]
elif ml_type == 'nodrug':
	features = DV.feature_names_ + ['age','n_i','n_ij']

# X and y
X = np.hstack((X_cat,X_num,X_bin))
#X = X[1:1000, :]
#y = dfML['n_ij_ddi'].values
y = dfML['n_ij_ddi'].apply(lambda x: 1 if x>0 else 0).values
#y = y[1:1000]

n_patients, n_features = X.shape
n_positive = sum([True for d in y if d ==1])
n_negative = sum([True for d in y if d ==0])
print 'Patients: {:,d}'.format(n_patients)
print 'Features: {:,d}'.format(n_features)
print
print 'Positives: {:,d} ({:.2%})'.format( n_positive, n_positive/n_patients)
print 'Negatives: {:,d} ({:.2%})'.format( n_negative, n_negative/n_patients)
print 


random_state = 2

classifiers = OrderedDict([
	('Uniform Dummy',DummyClassifier(strategy='uniform', random_state=random_state)),
	('Biased Dummy',DummyClassifier(strategy='stratified', random_state=random_state)),
	('Rough Dummy',RoughClassifier()),
	('Linear SVM',LinearSVC()),
	('Logistic Regression',LogisticRegression()),
])

FourFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)


# Classifier
print '-- Manual Fitting --'
r = list()
t = list()
f = list()

for clf_name, clf in classifiers.items():
	print '- Fitting: %s' % (clf_name)
	i = 1
	mean_precision = 0.0
	mean_recall = np.linspace(0, 1, 100)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0,1,100)

	for idx_train, idx_test in FourFold.split(X, y):
		print 'Split Fold: %d' % (i)
		X_train, y_train, X_test, y_test = X[idx_train], y[idx_train], X[idx_test], y[idx_test]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)

		if hasattr(clf, 'predict_proba'):
			probas_pred = clf.predict_proba(X_test)[:, 1]
		else:
			probas_pred = clf.decision_function(X_test)
			#probas_pred = (probas_pred - probas_pred.min()) / (probas_pred.max() - probas_pred.min())

		precisions, recalls, thresholds = precision_recall_curve(y_test, probas_pred, pos_label=1)
		fprs, tprs, thresholds = roc_curve(y_test, probas_pred, pos_label=1)

		pr_auc = auc(recalls, precisions)
		roc_auc = auc(fprs,tprs)
	
		if not 'Dummy' in clf_name:
			# To CSV
			for y_test_, probas_pred_ in zip(y_test, probas_pred):
				t.append((clf_name,i,y_test_,probas_pred_))

		# Eval
		precision, recall, f1, support = precision_recall_fscore_support(y_test,y_pred, average='binary', pos_label=1)
		mcc = matthews_corrcoef(y_test, y_pred)
		
		# Features
		if hasattr(clf, 'coef_'):
			coefs = clf.coef_ if len(clf.coef_)>1 else clf.coef_[0]
			for feature,coef in zip(features,coefs):
				f.append((clf_name,i,feature,coef))

		# Results
		r.append((clf_name,i,precision,recall, f1, support, mcc, roc_auc, pr_auc))
		i += 1


plt.suptitle('Precision & Recall (PR) and Receiver Operating Characteristic (ROC) curves')
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.97, top=0.90, wspace=0.08, hspace=0.18)
plt.savefig('images/img-ml-auc-%s.png' % (suffix)) #, dpi=100, frameon=True, bbox_inches='tight', pad_inches=0.4)


print '--- Save CSV ---'
if ml_type == 'simple':

	print 'AUC Thresholds'
	dfT = pd.DataFrame(t, columns=['clf','fold','y_test','probas'])
	with gzip.open('csv/ml_thresholds_%s.csv.gz' % (suffix), 'wb') as zfile:
		zfile.write(dfT.to_csv(encoding='utf-8', index=False))

	print 'Features'
	dfF = pd.DataFrame(f, columns=['clf','fold','feature','coef'])
	dfF = dfF.groupby(['clf','feature']).agg({'coef':'mean'})
	dfF.to_csv('csv/ml_features_%s.csv' % (suffix), encoding='utf-8', index=True)


print 'Classification'
dfR = pd.DataFrame(r, columns=['clf','fold','precision','recall','f1','support','mcc','roc_auc','pr_auc'])
dfR.sort_values('clf', inplace=True)
dfR.to_csv('csv/ml_results_%s.csv' % (suffix),  encoding='utf-8', index=False)
print dfR.groupby('clf').agg('mean')

