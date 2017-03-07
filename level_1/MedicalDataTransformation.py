# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:11:43 2017

@author: alexandernelson
"""

# libraries we will need for this notebook's experiment
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score   # an accuracy scoring function from scikit learn
from sklearn.model_selection import GridSearchCV

medical_data = pd.read_csv("datasets/balanced_cleaned_diabetes_data.csv")

# load in a transformed version of the dataset - yours should look something like this
medical_data_dummy = pd.get_dummies(medical_data['gender'])

df_new = pd.concat([medical_data_dummy, medical_data], axis=1)
df_new = df_new.drop('gender', 1)

age_replace = {'[0-10)': 0,  '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}
df_new['age'] = df_new['age'].replace(age_replace)
readmitted_replace = {'>30': -1, 'NO': -1, '<30': 1}
df_new['readmitted'] = df_new['readmitted'].replace(readmitted_replace)

transformed_medical_data = df_new
print(transformed_medical_data)

input_data_1 = transformed_medical_data[['Male','Female','age','number_emergency']]
labels_1 = transformed_medical_data['readmitted']

# TODO: import your tree-based algorithm and make a default instance of it
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

# range of parameters to test with cross-validation
parameters = {'max_depth': np.arange(1,10)}

# create an instance of the GridSearchCV cross-validator - using our classifier and choice or parameters
cross_validator = GridSearchCV(classifier, parameters)

# get the best parameter choice based on cross-validation and print out
cross_validator.fit(input_data_1,labels_1)        
best_param =  cross_validator.best_estimator_.max_depth     
print('Best parameter according to cross-validation param is = ' + str(best_param))

classifier = DecisionTreeClassifier(max_depth=best_param)

# fit our chosen classifier to the dataset
classifier.fit(input_data_1, labels_1)                              

# print out the number of misclassified points
predicted_labels = classifier.predict(input_data_1)
acc = len(labels_1) - accuracy_score(labels_1.ravel(), predicted_labels.ravel(), normalize=False)
print('Our classifier mislabeled ' + str(acc) + ' of ' + str(len(labels_1)) + ' points')

# load in SVC - a kernel classifier function from the scikit-learn library
from sklearn.svm import SVC

# create a default instance of the classifier
classifier = SVC()

# create a parameter range to test over
parameters = {'kernel':['rbf'], 'gamma':np.linspace(0,10,100)}

# create an instance of the GridSearchCV cross-validator - using our classifier and choice or parameters
cross_validator = GridSearchCV(classifier, parameters)

# get the best parameter choice based on cross-validation and print out
cross_validator.fit(input_data_1,labels_1)        
best_param =  cross_validator.best_estimator_.gamma     
print('Best parameter according to cross-validation is = ' + str(best_param))

# create an instance of a kernel-based regressor from scikit learn
classifier = SVC(kernel = 'rbf',gamma = best_param)

# fit our chosen classifier to the dataset
classifier.fit(input_data_1,labels_1)                              

# print out the number of misclassified points
predicted_labels = classifier.predict(input_data_1)
acc = len(labels_1) - accuracy_score(labels_1.ravel(), predicted_labels.ravel(), normalize=False)
print('Our classifier mislabeled ' + str(acc) + ' of ' + str(len(labels_1)) + ' points')
print('Finished')
