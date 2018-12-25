#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:10:09 2017

@author: ida
"""
import numpy as np
from lab7_helper import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


########## TOY DATASET ##########

categories = {0: 'furniture', 1:'restaurant'}

train_y_data = np.array([1,1,1,0,0])
train_x_data = ['our restaurants standards are high our ingredients are cooked fresh served fresh right to your table',
              'today there are over 300 restaurants in the UK serving fresh and unique flavours that will amaze you',
              'in our restaurants the worlds most exciting cuisines come from east asia where food is fresh healthy and delicious',
              'our furniture store is owned by ABC international and as such we have the infrastructure of a huge global furniture leader',
              'we have beautiful hardwood furniture to suit every taste including 20 exclusive ranges which feature natural oak rustic oak country painted vintage painted and dark stained acacia with more than 25 pieces of furniture in each collection']
train_x_pos = ['PRP NNS NNS VBP JJ PRP NNS VBP VBN JJ VBD JJ NN TO PRP NN',
               'NN EX VBP IN CD NNS IN DT NNP VBG JJ CC JJ NNS WDT MD VB PRP',
               'IN PRP NNS DT NNS RBS JJ NNS VBP IN JJ NN WRB NN VBZ JJ JJ CC JJ',
               'PRP NN NN VBZ VBN IN NNP JJ CC IN JJ PRP VBP DT NN IN DT JJ JJ NN NN',
               'PRP VBP JJ NN NN TO NN DT NN VBG CD JJ NNS WDT VBP JJ NN JJ NN NN VBD JJ VBN CC JJ VBN NN IN JJR IN CD NNS IN NN IN DT NN']
train_x_lemma = ['we restaurant standard be high we ingredient be cook fresh serve fresh right to  your table',
                 'today there be over 300 restaurant in the UK serve fresh and unique flavour that will amaze you',
                 'in we restaurant the world most exciting cuisine come from east asia where food be fresh healthy and delicious',
                 'we furniture store be own by ABC international and as such we have the infastructure of a huge global furniture leader',
                 'we have beautifull hardwood furniture to suit every taste include 20 exclusive range which feature natural oak rustic oak country paint vintage paint and dark stain acacia with more than 25 piece of furniture in each collection']

test_y_data = [1,0,1,1]
test_x_data = ['our restaurants food is fresh and served to you with exciting flavours',
              'As we’ve grown we’ve become more than just furniture',
              'You can also now re-energise with a refreshing coffee or lovely lunch in our cafés or a relaxing meal in our restaurant',
              'Diners sit at rustic oak tables']
test_x_pos = ['PRP NNS NN VBZ JJ CC VBD TO PRP IN JJ NS',
              'IN PRP VBP VBN PRP VBP VBN RBR IN RB NN',
              'PRP MD RB RB VB IN DT JJ NN CC JJ NN IN PRP NNS CC DT NN NN IN PRP NN',
              'NNS VBP IN JJ NN NNS']
test_x_lemma = ['we restaurant food be fresh and serve to you with exciting flavour',
                "As we've grow we've became more than just furniture",
                'You can also now re-energise with a refreshing coffee or lovely lunch in we café or a relaxing meal in we Butterfly Inn restaurant at Tillicoultry',
                'Diner sit at rustic oak table']


########## FEATURE EXTRACTION ##########

def vectorize(data):
    """
    Encoding the data into a matrix of word frequency counts.
    For each document d creates a vector of size 1x|V|, where V is the
    vocabulary of the whole data corpus. Each entry i in the vector
    represents the number of occurences of the word v_i in d.
    data: list of strings
    """
    # extract the vocabulary from the corpus
    # assume words in the documents are separated by whitespace
    vocab = list(set(' '.join(data).split()))
    encoded_data = encode_token_freq(data, vocab)
    return encoded_data, vocab

def encode_token_freq(data, vocab):
    """
    Encoding the data into a matrix of word frequency counts.
    For each document d creates a vector of size 1x|V|, where V is the
    vocabulary of the whole corpus. Each entry i in the vector
    represents the number of occurences of the word v_i in d.
    data: list of strings
    vocab: list of all word types occuring in the corpus
    """
    vectors = []
    for exp in data:
        # create a vector representation of the document
        # it has to be a numpy array
        # hint 1: you can use the toks.count(x) method of the list class,
        # which returns the number of occurences of x in list toks
        # hint 2: you may need to convert the counts from ints to floats to make your vector.
        exp_vector = np.zeros(len(vocab)) # students replace this
        vectors.append(exp_vector)
    return np.array(vectors)
    
train_word_features, word_vocab = vectorize(train_x_data)
train_pos_features, pos_vocab = vectorize(train_x_pos)
train_lemma_features, lemma_vocab = vectorize(train_x_lemma)

#test_word_features = encode_token_freq(test_x_data, word_vocab)

########## NAIVE BAYES ##########
def train_naive_bayes(data_x, data_y, alpha):
    return class_priors(data_y), feature_probabilities(data_x, data_y, alpha)

def class_priors(training_y):
    """
    estimate log prior probabilities for the classes
    training_y: labels of the training examples
    """
    classes, counts = np.unique(training_y, return_counts=True)
    log_probs = array_log(counts/sum(counts))
    priors = dict(zip(classes, log_probs))
    return priors

def feature_probabilities(training_x, training_y, alpha):
    """
    estimate log conditional probabilities of features given class
    training_x: vectorized training data
    training_y: labels for the training data
    alpha: smoothing parameter
    """
    num_exps, num_feat = training_x.shape
    f_probs = {}
    for c in set(training_y):       #for each category
        # c_exps is the list of indeces of training examples in category c
        c_exps = [i for i in range(0, num_exps) if training_y[i] == c]
        # c_data should be a numpy array of the size 1 x no.features
        # storing feature values for the whole category c
        # i.e. feature values summed over all documents in category c
        c_data = np.zeros(num_feat)     # students replace this
        c_data += alpha
        f_probs[c] = array_log(c_data/c_data.sum())
    return f_probs

def nb_classify(test_x, model):
    c_probs, f_probs = model
    classes = list(c_probs.keys())
    answers = []
    for exp in test_x:
        posteriors = [nb_posterior_log_probability(exp, c, c_probs, f_probs) for c in classes]
        answers.append(classes[np.argmax(posteriors)])
    return answers

def nb_posterior_log_probability(exp, c, c_probs, f_probs):
    """
    given a observation feature vector and a class, returns the log posterior
    probability of the class, i.e. log p(class|observation)
    exp: feature vector
    c: class id
    c_probs: class prior log probabilities
    f_probs: feature log probabilities conditional on class
    """
    posterior = 0     # students replace this
    return posterior
    

#nb_model = train_naive_bayes(train_word_features, train_y_data, 0.1)
#nb_classify(test_word_features, nb_model)

#nb_sk = MultinomialNB(alpha=0.1).fit(train_word_features, train_y_data)
#nb_sk.predict(test_word_features)

#most_probable_features(nb_model, word_vocab, categories, 3)


########## LOGISTIC REGRESSION ##########
#lr_sk = LogisticRegression().fit(train_word_features, train_y_data)
#lr_sk.predict(test_word_features)

#most_influential_features(lr_sk, word_vocab, categories, 6)

