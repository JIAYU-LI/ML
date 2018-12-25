#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:54:41 2017

@author: ida
"""
import math
import numpy as np
from sklearn.decomposition import PCA as PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

f = lambda x: math.log10(x)
array_log = np.vectorize(f)

def most_probable_features(model, vocab, categories, n):
    print("Highest probability features for:")
    for c, name in categories.items():
        print("\t class '{}':".format(name))
        if type(model) == sklearn.naive_bayes.MultinomialNB:
            feature_probs = model.feature_log_prob_[c]
        else:
            feature_probs = model[1][c][0]
        n_best = feature_probs.argsort()[-n:][::-1]
        n_best_words = [vocab[i] for i in n_best]
        print("\t" + ', '.join(n_best_words))
        print("\n")

def most_influential_features(model, vocab, categories, n):
    if len(categories) < 3:
        print("Most influential features for discriminating between classes:")
        coefs = np.fabs(model.coef_[0])
        n_best = coefs.argsort()[-n:][::-1]
        n_best_words = [vocab[i] for i in n_best]
        print("\t" + ', '.join(n_best_words))
    else:
        print("Most influential features for discriminating between:")
        for c, name in categories.items():
            print("\t class '{}' and others:".format(name))
            coefs = np.fabs(model.coef_[c])
            n_best = coefs.argsort()[-n:][::-1]
            n_best_words = [vocab[i] for i in n_best]
            print("\t" + ', '.join(n_best_words))
            print("\n")

########## DATA DISTRIBUTION ##########
# to plot graphs in a separate widow, change Spyder settings:
# Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
# then close and open Spyder.
colours = ['red', 'blue', 'lightgreen', 'purple']

def plot_data_lda(x_data, y_data, feature_names, categories, leg_loc=4):
    X_norm = (x_data - x_data.min())/(x_data.max() - x_data.min())
    lda = LDA(n_components=2)
    transformed = lda.fit_transform(X_norm, y_data)
    for i in categories:
        plt.scatter(transformed[y_data==i][:,0], transformed[y_data==i][:,1],
                    label=categories[i], c=colours[i], edgecolors='none')
    plt.legend(loc=leg_loc)
    plt.show()
#plot_data_lda(train_x_array, train_y, feature_names, categories, leg_loc=3)

def plot_data_pca(x_data, y_data, feature_names, categories, leg_loc=4):
    X_norm = (x_data - x_data.min())/(x_data.max() - x_data.min())
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(X_norm)
    for i in categories:
        plt.scatter(transformed[y_data==i][:,0], transformed[y_data==i][:,1],
                    label=categories[i], c=colours[i], edgecolors='none')
    plt.legend(loc=leg_loc)
    plt.show()

########## TRAIN CLASSIFIERS ##########
def nb_fit_and_predict(train_x_features, train_y_data,
                       test_x_features, test_y_data, average='binary'):
    nb = MultinomialNB(alpha=.1)
    nb.fit(train_x_features, train_y_data)
    nb_predict = nb.predict(test_x_features)
    print("\nNaive Bayes")
    print("test precision: {}".format(str(metrics.precision_score(test_y_data, nb_predict, average=average))))
    print("test recall: {}".format(str(metrics.recall_score(test_y_data, nb_predict, average=average))))
    print("test F1: {}".format(str(metrics.f1_score(test_y_data, nb_predict, average=average))))
    return nb, nb_predict

def lr_fit_and_predict(train_x_features, train_y_data,
                       test_x_features, test_y_data, average='binary'):
    lr = LogisticRegression(multi_class='multinomial',
                         solver='newton-cg')
    lr.fit(train_x_features, train_y_data)
    lr_predict = lr.predict(test_x_features)
    print("\nLogistic regression")
    print("test precision: {}".format(str(metrics.precision_score(test_y_data, lr_predict, average=average))))
    print("test recall: {}".format(str(metrics.recall_score(test_y_data, lr_predict, average=average))))
    print("test F1: {}".format(str(metrics.f1_score(test_y_data, lr_predict, average=average))))
    return lr, lr_predict

    
def confusion(test_y_data, predictions, categories):
    print("confusion matrix:")
    cm = confusion_matrix(test_y_data, predictions)     # Compare the gold standard to our predictions
    print (list(categories.values()))             # Remind ourselves of the classes
    print (cm) 