import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm

from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# this method selects the best parameter used in SMV classifier
def svc_param_selection(X, y, nfolds):
    Cs = [64, 128, 256, 512]
    gammas = [0.0001, 0.001, 0.01, 0.1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_
    

def process_images(images, labels):
    fabric = "cotton"

    x = []
    y = []
    WIDTH = 128
    HEIGHT = 128

    for img in images:
        base = os.path.basename(img)
        finding = labels["Type"][labels["Name"] == base].values[0]

        full_size_image = cv2.imread(img, 0)
        x.append(cv2.resize(full_size_image,(WIDTH,HEIGHT)))

        if fabric in finding:
            finding = 1
            y.append(finding)
        else:
            finding = 0
            y.append(finding)
        
    return np.array(x), np.array(y)   


def reshape_matrices_to_vectors(x):
    image_vectors = []

    for matrix in x:
        image_vectors.append(np.reshape(matrix, 128 * 128))

    return np.array(image_vectors)

# this method scales the data set so that all values in the dataset are
# in a specific range and the classifier is not biased towards extreme values
def scale_data(X):
    min_max_scaler =  MinMaxScaler()
    scaled_X = min_max_scaler.fit_transform(X)

    return scaled_X

# this method extract important features from the image matrices using
# tree based classifier. After the features are selected the index of the
# features that are selected is stored in a list. These indices are necessary
# to keep track of which features are selected and important. Testing data needs 
# to be formatted with these indices so that tha training and testing data 
# dimensions are same
def feature_selection(X, y):

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    indices = model.get_support()
    feature_indices = []
    for i in range(len(indices)):
        if indices[i] == True:
            feature_indices.append(i)

    return X_new, feature_indices

# testing data is formatted with the indices that are selected from the feature selection
# step. Only the selected feature indices are kept and other features are discarded
def format_test_data(X, feature_indices):
    return X[:, feature_indices]

# this method takes numerical matrices converted from image and pefomr feature selection
# and process the dataset and returns training/testing data
def feature_selection_and_data_building(train_x, test_x, train_y, test_y):                                                                                                                                                                                                                                                                                        
    # this is calling matrices_to_vectors
    # this method takes the matrices and reshape them into vectors and returns it
    reshaped_train_x = reshape_matrices_to_vectors(train_x)
    reshaped_test_x = reshape_matrices_to_vectors(test_x)

    # this method scales the dataset. Scaling is important since the value can have a big range
    # scaling them into a certain range is important so that data is generalized and every 
    # feature has same level of imporatnce. If scaling is not done then some feature may have 
    # unfair advantage because of their large range of values.
    scaled_train_X = scale_data(reshaped_train_x)
    scaled_test_X = scale_data(reshaped_test_x)

    # this is feature selection method. This method uses tree-based feature selection to 
    # select important features. The method takes the scaled data and perform feature 
    # selection on the data
    Training_X, feature_indices = feature_selection(scaled_train_X, train_y)
    Training_Y = train_y

    Testing_X = format_test_data(scaled_test_X, feature_indices)
    Testing_Y = test_y

    return Training_X, Training_Y, Testing_X, Testing_Y, len(feature_indices)

# this method is the classifier. Here SVM is used with a default parameter
def classify_image(Training_X, Training_Y, Testing_X, best_c, best_gamma):
    clf = SVC(C=best_c, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=best_gamma, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
    clf.fit(Training_X, Training_Y)
    Predicted_Y = clf.predict(Testing_X)

    return Predicted_Y

# path of the directory that sotres the images
path = "images"

# file to store results
fopen = open('result.txt', 'w')
fopen.write('Iteration' + '\t' + '#features' + '\t' + 'Accuracy' + '\t' + 'Sensitivity' + '\t' + 'Specificity' + '\t' + 'CorrectlyIdentifiedCotton' + '\t' + 'WronglyIdentifiedCotton' + 
    '\t' + 'CorrectlyIdentifiedPoly' + '\t' + 'WronglyIdentifiedPoly' + '\n')

# read images form the image folder
images = glob(os.path.join(path, "*.jpg"))
labels = pd.read_csv('Labels.csv')

# process images. Convert images into number matrix
X, Y = process_images(images, labels)

# spliting the dataset into random training/testing data
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# select parameters for SVM
Training_X, Training_Y, Testing_X, Testing_Y, num_feat = feature_selection_and_data_building(train_x, test_x, train_y, test_y)

# 10-fold cross validation to find the best parameters for SVM
best_params, best_cv_score = svc_param_selection(Training_X, Training_Y, 10)
best_c = best_params['C']
best_gamma = best_params['gamma']

# run experiments 10 times
for exp_i in range(0, 10):
    print "Experiment: " + str(exp_i + 1)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.33, random_state = 42)
    Training_X, Training_Y, Testing_X, Testing_Y, num_feat = feature_selection_and_data_building(train_x, test_x, train_y, test_y)

    # classifier
    Predictedd_Y = classify_image(Training_X, Training_Y, Testing_X, best_c, best_gamma)
    acc = accuracy_score(Testing_Y, Predictedd_Y)

    TP, TN, FP, FN = 0,0,0,0
    for i in range(len(Predictedd_Y)):
        if Testing_Y[i] == Predictedd_Y[i] == 1:
            TP += 1
        if Predictedd_Y[i] == 1 and Testing_Y[i] != Predictedd_Y[i]:
            FP += 1
        if Testing_Y[i] == Predictedd_Y[i] == 0:
            TN += 1
        if Predictedd_Y[i] == 0 and Testing_Y[i] != Predictedd_Y[i]:
            FN += 1

    sensitivity = float(TP)/float(TP + FP)
    specificity = float(TN)/float(TN + FP)

    fopen.write(str(exp_i + 1) + '\t' + str(num_feat) + '\t' + str(acc) + '\t' + str(sensitivity) + '\t' + str(specificity) + '\t' + str(TP) + '\t' + str(FP) + '\t' + str(TN) + '\t' + str(FN) + '\n')


    for i in Predictedd_Y:
        if (i == 1):
            print "Cotton"
        else :
            print "Polyster"

fopen.write('\n')
fopen.write('Parameter used for SVM:' + '\n')
fopen.write('C: ' + str(best_c) + ' and gamma:' + str(best_gamma) + '\n')
fopen.write('Best score after 10-fold cross validation: ' + str(best_cv_score) + '\n')

fopen.write("Total number of training data: " + str(Training_X.shape[0]) + '\n')
fopen.write("Total number of testing data: " + str(Testing_X.shape[0]) + '\n')

