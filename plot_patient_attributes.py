# coding=utf-8
# Author: Rion B Correia
# Date: Nov 16, 2014
#
# Description: Plot CoAdmin and DDI Distribution 
#
from __future__ import division
import matplotlib as mpl
import matplotlib.style
mpl.style.use('classic')
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

import numpy as np
import scipy
from scipy import interp
from scipy.stats import ttest_ind
import pandas as pd
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)
import util

#
#
# Load CSVs
#
dfu, dfc, dfi = util.dfUsersInteractionsSummary(loadCoAdmin=False)

print '>> dfu'
print dfu.head()


ind = np.arange(10**3)
qt_coadmins = []
qt_inter = []


fig = plt.figure(figsize=(10,2.7))
renderer = fig.canvas.get_renderer()
ax1 = plt.subplot(1,1,1)
plt.rc('font', size=12)
plt.rc('legend', fontsize=12)
plt.rc('legend', numpoints=1)


width = 0.85

total = len(dfu.index.unique())

# Gender
males = len( dfu.loc[ (dfu['gender']=='Male'), : ].index.unique() )
females = len( dfu.loc[ (dfu['gender']=='Female'), : ].index.unique() )
left = 0
for label,value,color in zip(['Male','Female'],[males,females],['blue','red']):
	percent = value/total
	b = ax1.barh(2, percent, width, color=color, left=left, alpha=0.5)
	#
	patch = b.get_children()[0]
	bx,by = patch.get_xy()
	tx,ty = 0.5*patch.get_width() + bx , 0.43*patch.get_height() + by
	#
	ax1.text(tx,ty, label, ha='center', va='center', rotation=0)
	#
	left += percent

# Age
left = 0
age_group = dfu['age_group'].value_counts()
age_group = age_group.sort_index()
print age_group
print age_group/age_group.sum()
color = iter( cm.jet_r(np.linspace(0,1,age_group.shape[0]+1)) )
for label,value in age_group.iteritems():
	percent = value/total
	b = ax1.barh(1, percent, width, color=next(color), left=left, alpha=0.5)
	#
	patch = b.get_children()[0]
	bx,by = patch.get_xy()
	tx,ty = 0.5*patch.get_width() + bx , 0.52*patch.get_height() + by
	#
	if label not in ['80-84', '85-89', '90-94', '95-99', '>99']:
		ax1.text(tx,ty, label, ha='center', va='center', rotation=90)
	#
	left += percent


# Education
left = 0
escolaridade = dfu['education'].value_counts()
escolaridade = escolaridade.sort_index()
left_annotate = -0.075
cmcolor = iter( cm.Paired(np.linspace(0,1,escolaridade.shape[0])) )
#next(cmcolor)
for label,value in escolaridade.iteritems():
	percent = value/total
	color = 'gray' if (label=='Not reported') else next(cmcolor)
	b = ax1.barh(0, percent, width, color=color, left=left, alpha=0.5)
	#
	patch = b.get_children()[0]
	bx,by = patch.get_xy()
	tx,ty = 0.5*patch.get_width() + bx , 0.47*patch.get_height() + by
	#
	if label in ['Cant read/write','Complete elementary','Complete high school']:
		if label == 'Cant read/write':
			shortlabel = 'Illiter.'
		elif label == 'Complete elementary':
			shortlabel = 'K-6'
		elif label == 'Complete high school':
			shortlabel = 'K-12'
		ty = 0.52*patch.get_height() + by
		ax1.text(tx,ty, shortlabel, ha='center', va='center', rotation=90)
	if label in ['Not reported','Incomplete elementary']:
		label = label.replace(' ','\n')
		ax1.text(tx,ty, label, ha='center', va='center', rotation=0)
	if label in ['Can read/write a note','Incomplete high school','Incomplete college','Complete college']:
		ax, ay = (left_annotate) , -0.6
		al = ax1.annotate(label, xy=(tx, 0.25*patch.get_height()+by), xycoords='data', xytext=(ax, ay),
			arrowprops=dict(facecolor='black', color="0.5", arrowstyle="<|-,head_length=0.3,head_width=0.15",
							connectionstyle="angle3,angleA=0,angleB=90"),
			horizontalalignment='left', verticalalignment='middle'
			)
		left_annotate += 0.235
	#
	left += percent

ax1.set_title('Patient distribution')



#
#
#
xticks = np.linspace(0,1,11,endpoint=True)
xticklabels = ['%.1f' % x for x in xticks]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticklabels)

yticks = np.array([0,1,2]) + (width/2)
ax1.set_yticks(yticks)
ax1.set_yticklabels(['Education','Age','Gender'])

ax1.set_xlim(0,1)
ax1.set_ylim(0,2.8)
#ax1.grid()

plt.subplots_adjust(left=0.10, bottom=0.18, right=0.98, top=0.89, wspace=0.00, hspace=0.00)
plt.savefig('images/img-patient-attrib.pdf', dpi=300, frameon=True, bbox_inches=None, pad_inches=0.0)


