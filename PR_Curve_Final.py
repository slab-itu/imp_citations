# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 14:03:32 2016


"""
import numpy
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.metrics import auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# import some data to play with
mydata = pd.read_csv("C:/Users/Asma/Desktop/FINAL_TABLE_USER2 - Copy - Copy.csv")
y = mydata["TOP_25_PAPERS"]  #provided your csv has header row, and the label column is named "Label"
n_points=len(mydata)
##select all but the last column as data
X = mydata.ix[:,:-1]
X=X.iloc[:,:]

##################################


cv = StratifiedKFold(y, n_folds=10)

y_real = []
y_proba = []

#classifier = svm.SVC(kernel='rbf',gamma=0.001, C=100, probability=True, class_weight ='balanced')
classifier = RandomForestClassifier(n_estimators=100,
                                 class_weight="auto",
                                 criterion='gini',
                                 bootstrap=True,
                                 max_features=0.5,
                                 min_samples_split=1,
                                 min_samples_leaf=5,
                                 max_depth=10,
                                 n_jobs=1)

#classifier= KNeighborsClassifier()
#classifier = GaussianNB()
#classifier = DecisionTreeClassifier(max_depth=15)

for i, (train, test) in enumerate(cv):
    x_train=X[train[0]:train[len(train)-1]]
    x_test=X[test[0]:test[len(test)-1]]
    y_train= y[train[0]:train[len(train)-1]]
    y_test=y[test[0]:test[len(test)-1]]
             
    probas_ = classifier.fit(x_train, y_train).predict_proba( x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, probas_[:, 1])
    lab = 'Pre-Recall fold %d (area = %0.2f)' % (i+1, auc(recall, precision))
    plt.plot(recall, precision, lw=1, label=lab)

    y_real.append(y_test)
    y_proba.append(probas_[:, 1])
    
    
    
y_real = numpy.concatenate(y_real)
y_proba = numpy.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Mean Pre-Recall (area = %0.2f)' % (auc(recall, precision))
plt.plot(recall, precision, label=lab, lw=2, color='black') 
plt.xlim([0.02, 0.99])
plt.ylim([0, 1.05])
plt.grid(True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve, Random Forest')
plt.rcParams['axes.facecolor']='white'
plt.legend(loc="lower left")
plt.show()


    
    
    