# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:03:32 2016

@author: anam
"""
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
###############################################################################


# import some data 
mydata = pd.read_csv("C:/Users/Asma/Desktop/FINAL_TABLE_USER2 - Copy - Copy.csv")
y = mydata["TOP_25_PAPERS"]  #provided your csv has header row, and the label column is named "Label")
##select all but the last column as data
X = mydata.ix[:,:-1]
X=X.iloc[:,:]

############################################################
## Build a forest and compute the feature importances
#forest = ExtraTreesClassifier(n_estimators=100,
#                              random_state=0)
#
#forest.fit(X1, y)
#
#model = SelectFromModel(forest, prefit=True)
#X = model.transform(X1)

######################

# Classification and ROC analysis
# Run SVM/Random forest classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(y, n_folds=10)

#classifier = svm.SVC(kernel='rbf',gamma=0.001, C=100, probability=True, class_weight ='balanced')
#classifier = RandomForestClassifier(n_estimators=50,
#                                 class_weight="auto",
#                                 criterion='gini',
#                                 bootstrap=True,
#                                 max_features=0.5,
#                                 min_samples_split=1,
#                                 min_samples_leaf=10,
#                                 max_depth=10,
#                                 n_jobs=1)

#classifier= KNeighborsClassifier()
#classifier = GaussianNB()
classifier = DecisionTreeClassifier(max_depth=8)
################  ROC analysis

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    x_train=X[train[0]:train[len(train)-1]]
    x_test=X[test[0]:test[len(test)-1]]
    y_train= y[train[0]:train[len(train)-1]]
    y_test=y[test[0]:test[len(test)-1]]
    
    
    probas_ = classifier.fit(x_train, y_train).predict_proba( x_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], color=(0.6, 0.6, 0.6))

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr , 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.02, 1.0])
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic, Decision Tree')
plt.legend(loc="lower right")
plt.show()