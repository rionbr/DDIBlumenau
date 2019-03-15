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
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, roc_curve, precision_recall_curve, auc
from itertools import cycle


if __name__ == '__main__':

	suffix = 'simple'
	dfT = pd.read_csv('csv/ml_thresholds_%s.csv.gz' % (suffix), encoding='utf-8')


	fig, axes = plt.subplots(2,2,figsize=(8,8), sharex=True, sharey=False)
	plt.rc('font', size=12)
	plt.rc('legend', fontsize=10)
	plt.rc('legend', numpoints=1)
	plt.rc('legend', labelspacing=0.3)
	colors = cycle(['red','blue','darkorange','magenta'])


	print dfT.head()
	i_clf = 0
	i_fold = 0
	#axes = [ax1,ax2,ax3,ax4]

	for clf, dfTc in dfT.groupby('clf'):
		print '> clf: %s (%d)' % (clf, i_clf)
	
		i_fold = 0

		mean_precision = 0.0
		mean_recall = np.linspace(0, 1, 100)
		mean_tpr = 0.0
		mean_fpr = np.linspace(0,1,100)

		axPR = axes[0][i_clf]
		axROC = axes[1][i_clf]

		axPR.set(adjustable='box-forced', aspect='equal') # equal weight and height
		axROC.set(adjustable='box-forced', aspect='equal') # equal weight and height

		for fold, dfTcf in dfTc.groupby('fold'):
			print '> fold: %s' % fold

			color = next(colors)

			
			y_test = dfTcf['y_test']
			probas_pred = dfTcf['probas']

			pos_neg = y_test.value_counts()
			n_positive = pos_neg.loc[1]
			n_negative = pos_neg.loc[0]

			precision, recall, thresholds = precision_recall_curve(y_test, probas_pred, pos_label=1)
			fpr, tpr, thresholds = roc_curve(y_test, probas_pred, pos_label=1)

			pr_auc = auc(recall, precision)		
			roc_auc = auc(fpr,tpr)
			
			precision = precision[::-1]
			recall = recall[::-1]
			thresholds = thresholds[::-1]

			mean_precision += interp(mean_recall, recall, precision)
			mean_tpr += interp(mean_fpr, fpr, tpr)
			mean_tpr[0] = 0.0

			axPR.plot(recall, precision, lw=2, color=color, label='P/R fold %d (area=%.2f)' % (i_fold, pr_auc), zorder=2)
			axROC.plot(fpr, tpr, lw=2, color=color, label='ROC fold %d (area=%.2f)' % (i_fold, roc_auc), zorder=2)
		
			i_fold += 1
		i_clf += 1

		mean_precision /= 4
		mean_pr_auc = auc(mean_recall, mean_precision)
		axPR.plot(mean_recall, mean_precision, color='g', linestyle='--', label='Mean Int. P/R (area=%0.2f)' % mean_pr_auc, lw=2, zorder=5)

		# Luck Lines
		pr_baseline = (n_positive/float(n_positive+n_negative))
		axPR.plot([0, 1], [pr_baseline, pr_baseline], linestyle='--', lw=2, color='k', label='Baseline')
		axROC.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Baseline')


		mean_tpr /= 4
		mean_tpr[-1] = 1.0
		mean_roc_auc = auc(mean_fpr, mean_tpr)
		axROC.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean Int. ROC (area=%0.2f)' % mean_roc_auc, lw=2, zorder=5)

		for ax in [axPR,axROC]:
			ax.set_xlim([-0.05, 1.05])
			ax.set_ylim([-0.05, 1.05])
			ax.grid()

			
		axPR.set_xlabel('Recall')
		axPR.set_ylabel('Precision')
		axROC.set_xlabel('False Positive Rate')
		axROC.set_ylabel('True Positive Rate')

		axPR.set_title(clf)
		axPR.legend(loc="lower left")
		axROC.legend(loc="lower right")

		i_fold += 1


	#plt.suptitle('Precision & Recall (P/R) and Receiver Operating Characteristic (ROC) curves')
	plt.subplots_adjust(left=0.07, bottom=0.07, right=0.99, top=0.94, wspace=0.16, hspace=0.16)
	#plt.tight_layout()
	plt.savefig('images/img-ml-auc-%s.pdf' % (suffix), dpi=300, pad_inches=0.0)


