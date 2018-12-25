#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:25:47 2017

@author: ida
"""
from lab7_helper import *
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import normalize


########## BIG DATASET ##########


data = np.load("/afs/inf.ed.ac.uk/group/teaching/anlp/lab7/complaints_text.npy")
categories = {0: 'Student loan', 1: 'Consumer Loan',
                  2: 'Bank service', 3: 'Credit card'}
all_x_data = list(zip(data[:,1], data[:,2], data[:,3]))
all_y_data = data[:,0].astype("int64")

train_x, test_x, train_y, test_y = train_test_split(all_x_data,
                                                    all_y_data,
                                                    stratify=all_y_data,
                                                    test_size=.20,
                                                    random_state=42)
train_x_words, train_x_pos, train_x_lemma = zip(*train_x)
test_x_words, test_x_pos, test_x_lemma = zip(*test_x)



########## FEATURE EXTRACTION ##########


# word counts
vectorizer = CountVectorizer()
train_x_features = vectorizer.fit_transform(train_x_words)
test_x_features = vectorizer.transform(test_x_words)
feature_names = vectorizer.get_feature_names()

print ("Initial method extracted a total of", len(feature_names), "features")

        

########## TRAIN CLASSIFIERS ##########


#nb_model, nb_predict = nb_fit_and_predict(train_x_features, train_y, test_x_features,
#                                test_y, average='weighted')
#
#lr_model, lr_predict = lr_fit_and_predict(train_x_features, train_y, test_x_features,
#                                test_y, average='weighted')
#
#most_probable_features(nb_model, feature_names, categories, 10)
#most_influential_features(lr_model, feature_names, categories, 10)
#
## We can use a confusion matrix to see which categories are mistaken for which others.
## cell [i,j] shows how many docs from class i were labelled as class j by the model.
#confusion(test_y, nb_predict, categories)
#confusion(test_y, lr_predict, categories)


        
########## DIMENSIONALITY REDUCTION ##########

 
### word counts excluding stopwords

#print("\n***Filtering out stopwords***")
#stopset = set(stopwords.words('english'))
#vectorizer_stop = CountVectorizer(stop_words=stopset)
#train_x_features_stop = vectorizer_stop.fit_transform(train_x_words)
#test_x_features_stop = vectorizer_stop.transform(test_x_words)
#feature_names_stop = vectorizer_stop.get_feature_names()
#print ("Without stopwords extracted a total of", len(feature_names_stop), "features")
#nb_stop_model, nb_stop_predict = nb_fit_and_predict(train_x_features_stop, train_y,
#                                                    test_x_features_stop, test_y,
#                                                    average='weighted')
#lr_stop_model, lr_stop_predict = lr_fit_and_predict(train_x_features_stop, train_y,
#                                                    test_x_features_stop, test_y,
#                                                    average='weighted')
#most_probable_features(nb_stop_model, feature_names_stop, categories, 10)
#most_influential_features(lr_stop_model, feature_names_stop, categories, 10)


### remove low-variance features

#print("\n***Removing low variance features***")
#vt = VarianceThreshold(threshold=0.01)
#train_x_features_vt = vt.fit_transform(train_x_features)
#test_x_features_vt = vt.transform(test_x_features)
#feature_names_vt = [f for (f, b) in zip(feature_names, vt.get_support()) if b]
#print ("Without low-variance features extracted a total of", len(feature_names_vt), "features")
#nb_vt_model, nb_vt_predict = nb_fit_and_predict(train_x_features_vt, train_y,
#                                                test_x_features_vt, test_y,
#                                                average='weighted')
#lr_vt_model, lr_vt_predict = lr_fit_and_predict(train_x_features_vt, train_y,
#                                                test_x_features_vt, test_y,
#                                                average='weighted')
#most_probable_features(nb_vt_model, feature_names_vt, categories, 10)
#most_influential_features(lr_vt_model, feature_names_vt, categories, 10)


### normalize
def plot_document_lengths(corpus):
    lengths = {}
    for d in corpus:
        if len(d.split()) not in lengths:
            lengths[len(d.split())] = 1
        else:
            lengths[len(d.split())] += 1
    plt.bar(list(lengths.keys()), list(lengths.values()), width = 1.0,
            linewidth=0, color='red')
    plt.ylabel('frequency')
    plt.xlabel('number of words')
    
#plot_document_lengths(train_x_words)
#print("\n***Normalizing feature vectors***")
#train_x_normalized = normalize(train_x_features_vt, norm='l2')
#test_x_normalized = normalize(test_x_features_vt, norm='l2')
#nb_vt_model, nb_vt_predict = nb_fit_and_predict(train_x_normalized, train_y,
#                                                test_x_normalized, test_y,
#                                                average='weighted')
#lr_vt_model, lr_vt_predict = lr_fit_and_predict(train_x_normalized, train_y,
#                                                test_x_normalized, test_y,
#                                                average='weighted')


########## FURTHER EXPERIMENTS ##########
# vectorization options

# binary word occurence features
#vectorizer = CountVectorizer(stop_words=stopset, binary=True)

# ngrams
# you can include only ngrams of size n (ngram_range = (n, n))
# or ngrams of different sizes, e.g. unigrams, bigrams, and trigrams (ngram_range = (1, 3))
#vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b',
#                             stop_words=stopset)

# binary ngrams
#vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b',
#                             stop_words=stopset, binary=True)

# in tf-idf space
#vectorizer = TfidfVectorizer(ngram_range=(1,3), token_pattern=r'\b\w+\b',
#                             stop_words=stopset)
