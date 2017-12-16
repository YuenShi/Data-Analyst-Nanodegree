#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#   Data Example:
#   {'METTS MARK': {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN', 'total_payments': 1061827, 'exercised_stock_options': 'NaN', 'bonus': 600000, 'restricted_stock': 585062, 'shared_receipt_with_poi': 702, 'restricted_stock_deferred': 'NaN', 'total_stock_value': 585062, 'expenses': 94299, 'loan_advances': 'NaN', 'from_messages': 29, 'other': 1740, 'from_this_person_to_poi': 1, 'poi': False, 'director_fees': 'NaN', 'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'mark.metts@enron.com', 'from_poi_to_this_person': 38}

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person']
#[ True  True  True  True  True False  True False False
# False False False False  True False False]
#features_list = ['poi','salary','to_messages','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','total_stock_value','from_messages','from_this_person_to_poi','deferred_income','from_poi_to_this_person']
# You will need to use more features
#features_list = ['poi','salary','to_messages','total_payments','exercised_stock_options','shared_receipt_with_poi','expenses','loan_advances','from_messages','from_this_person_to_poi','from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#    print data_dict

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


### Task 3: Create new feature(s)
import copy
data_dict_new = copy.deepcopy(data_dict)
for key in data_dict:
    if (data_dict[key]['to_messages'] != 'NaN' and
        data_dict[key]['to_messages'] != 0 and
        data_dict[key]['from_poi_to_this_person'] != 'NaN' and
        data_dict[key]['from_poi_to_this_person'] != 0) :
        data_dict_new[key]['poi_to_percent'] = data_dict[key]['from_poi_to_this_person'] / float(data_dict[key]['to_messages'])
    else:
        data_dict_new[key]['poi_to_percent'] = 'NaN'
    if (data_dict[key]['from_messages'] != 'NaN' and
        data_dict[key]['from_messages'] != 0 and
        data_dict[key]['from_this_person_to_poi'] != 'NaN' and
        data_dict[key]['from_this_person_to_poi'] != 0) :
        data_dict_new[key]['poi_from_percent'] = data_dict[key]['from_this_person_to_poi'] / float(data_dict[key]['from_messages'])
    else:
        data_dict_new[key]['poi_from_percent'] = 'NaN'

features_list_new = copy.deepcopy(features_list)
features_list_new.append('poi_to_percent')
features_list_new.remove('to_messages')
features_list_new.remove('from_poi_to_this_person')
features_list_new.append('poi_from_percent')
features_list_new.remove('from_messages')
features_list_new.remove('from_this_person_to_poi')
### Store to my_dataset for easy export below.
use_data = 1
if use_data == 0:
    my_dataset = data_dict
    features_list = features_list
else:
    my_dataset = data_dict_new
    features_list = features_list_new

#print features_list
#max_k = len(features_list) - 1
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Example starting point. Try investigating other evaluation techniques!
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "======== GaussianNB ========"
print "Accuracy: ", clf.score(features_test,labels_test)
print "F1;", f1_score(labels_test,pred)
print "Precision:", precision_score(labels_test,pred)
print "Recall:", recall_score(labels_test, pred)

dt.fit(features_train, labels_train)
pred = dt.predict(features_test)
print "======== Decision Tree ========"
print "Accuracy: ", dt.score(features_test,labels_test)
print "F1;", f1_score(labels_test,pred)
print "Precision:", precision_score(labels_test,pred)
print "Recall:", recall_score(labels_test, pred)


# Use SelectKBest to select features
new_features_list = []
scores = []
from sklearn.feature_selection import SelectKBest
selection = SelectKBest(k=8)
selected_features = selection.fit_transform(features_train, labels_train)
for i in range(len(selection.get_support())):
    if selection.get_support()[i] == True:
        new_features_list.append(features_list[i+1])
        scores.append(selection.scores_[i])
print 'Features selected by SelectKBest:'
print new_features_list
print 'Selected features score:'
print scores

features_list = ['poi'] + new_features_list
new_features_list.remove('poi_from_percent')
features_list_without_create_feature = ['poi'] + new_features_list

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from time import time
from tester import test_classifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=42)
parameters = {'max_features':[2,4,6,7,8],'max_depth': [1,3,5],'min_samples_split':[2,4,6]}
#dt = DecisionTreeClassifier()
t0 = time()
grid_obj = GridSearchCV(dt, parameters,scoring = 'f1', cv=sss)
print "======== Decision Tree (Optimized) ========"
print("DecisionTree tuning: %r" % round(time()-t0, 3))
# TODO: Fit the grid search object to the training data and find the optimal parameters
t0 = time()
grid_obj = grid_obj.fit(features, labels)
print("DecisionTree fitting: %r" % round(time()-t0, 3))
# Get the estimator
dt = grid_obj.best_estimator_
## Print the parameters
print dt.get_params(), '\n'

print 'Result of feature_list without new create feature:'
test_classifier(dt, my_dataset, features_list_without_create_feature, folds = 100)

print 'Result of feature_list with new create feature:'
test_classifier(dt, my_dataset, features_list, folds = 100)


clf = dt
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
