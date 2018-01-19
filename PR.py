# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:43:05 2017

@author: Asma
"""

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
from sklearn.metrics import classification_report


# import some data to play with
file = 'our_model.csv'
print(file)
mydata = pd.read_csv("data/"+file)
y = mydata["Class"]  #provided your csv has header row, and the label column is named "Label"
n_points=len(mydata)
##select all but the last column as data
X = mydata.ix[:,:-1]
#X=X.iloc[:,:]

##################################


cv = StratifiedKFold(y, n_folds=7)

y_real1 = []
y_proba1 = []

y_real2 = []
y_proba2 = []

y_real3 = []
y_proba3 = []

y_real4 = []
y_proba4 = []

y_real5 = []
y_proba5 = []

classifier1 = svm.SVC(kernel='rbf',gamma=.04, C=1.00, probability=True, class_weight ='balanced')
classifier2 = RandomForestClassifier(n_estimators=50,
                                 class_weight="auto",
                                 criterion='gini',
                                 bootstrap=True,
                                 max_features=0.9,
                                 min_samples_split=1,
                                 min_samples_leaf=5,
                                 max_depth=11,
                                 n_jobs=1)

classifier3= KNeighborsClassifier()
classifier4 = GaussianNB()
classifier5 = DecisionTreeClassifier(max_depth=5)

for i, (train, test) in enumerate(cv):
    x_train=X[train[0]:train[len(train)-1]]
    x_test=X[test[0]:test[len(test)-1]]
    y_train= y[train[0]:train[len(train)-1]]
    y_test=y[test[0]:test[len(test)-1]]
    probas1_ = classifier1.fit(x_train, y_train).predict_proba( x_test)
    probas2_ = classifier2.fit(x_train, y_train).predict_proba( x_test)
    probas3_ = classifier3.fit(x_train, y_train).predict_proba( x_test)
    probas4_ = classifier4.fit(x_train, y_train).predict_proba( x_test)
    probas5_ = classifier5.fit(x_train, y_train).predict_proba( x_test)
    
    #prediction1 = classifier1.predict(x_test)
    #print(classification_report(y_test,prediction1))


    ##Random Forest Table
    #prediction = classifier2.predict(x_test)
    #print(classification_report(y_test,prediction))
    #fh = open('FinalResult2.txt', 'a') 
    #fh.write(Result+'\n') 
    
    #precision, recall, thresholds = precision_recall_curve(y_test, probas1_[:, 1])
    #lab = 'Pre-Recall fold %d (area = %0.2f)' % (i+1, auc(recall, precision))
    #plt.plot(recall, precision, lw=1, label=lab)

    y_real1.append(y_test)
    y_proba1.append(probas1_[:, 1])
    
    y_real2.append(y_test)
    y_proba2.append(probas2_[:, 1])
    
    y_real3.append(y_test)
    y_proba3.append(probas3_[:, 1])
    
    y_real4.append(y_test)
    y_proba4.append(probas4_[:, 1])
    
     
    y_real5.append(y_test)
    y_proba5.append(probas5_[:, 1])
    
y_real1 = numpy.concatenate(y_real1)
y_proba1 = numpy.concatenate(y_proba1)

y_real2 = numpy.concatenate(y_real2)
y_proba2 = numpy.concatenate(y_proba2)

y_real3 = numpy.concatenate(y_real3)
y_proba3 = numpy.concatenate(y_proba3)

y_real4 = numpy.concatenate(y_real4)
y_proba4 = numpy.concatenate(y_proba4)

y_real5 = numpy.concatenate(y_real5)
y_proba5 = numpy.concatenate(y_proba5)



##SVM Table
prediction = classifier1.predict(x_test)
print(classification_report(y_test,prediction))


##Random Forest Table
prediction = classifier2.predict(x_test)
print(classification_report(y_test,prediction))

precision1, recall1, _ = precision_recall_curve(y_real1, y_proba1)
precision2, recall2, _ = precision_recall_curve(y_real2, y_proba2)
precision3, recall3, _ = precision_recall_curve(y_real3, y_proba3)
precision4, recall4, _ = precision_recall_curve(y_real4, y_proba4)
precision5, recall5, _ = precision_recall_curve(y_real5, y_proba5)


lab = 'svm(area = 0.80)' % (auc(recall1, precision1))
lab2 = 'random forest(area = 0.80)' % (auc(recall2, precision2))
'''
lab = 'svm(area = %0.2f)' % (auc(recall1, precision1))
lab2 = 'random forest(area = %0.2f)' % (auc(recall2, precision2))
lab3 = 'KNN(area = %0.2f)' % (auc(recall3, precision3))
lab4 = 'Naive Bayes(area = %0.2f)' % (auc(recall4, precision4))
lab5 = 'Decision Tree(area = %0.2f)' % (auc(recall5, precision5))
'''

plt.plot(0, 0, label=lab, lw=1, color='blue') 
plt.plot(0, 0, label=lab2, lw=1, color='green') 

#plt.plot(recall1, precision1, label=lab, lw=1, color='blue') 
#plt.plot(recall2, precision2, label=lab2, lw=1, color='green') 
#plt.plot(recall3, precision3, label=lab3, lw=2, color='blue') 
#plt.plot(recall4, precision4, label=lab4, lw=2, color='purple') 
#plt.plot(recall5, precision5, label=lab5, lw=2, color='aqua') 

plt.xlim([0, 1.000000001])
plt.ylim([0, 1.05])
plt.grid(True)
plt.xlabel('recall')
plt.ylabel('precision')
#plt.title('Precision Recall curve')
plt.rcParams['axes.facecolor']='white'
plt.legend(loc="center left")
plt.show()


    
    
    