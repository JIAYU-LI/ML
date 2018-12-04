import nltk

#import brown corpus
from nltk.corpus import brown

# module for training a Hidden Markov Model and tagging sequences
from nltk.tag.hmm import HiddenMarkovModelTagger

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

# module for computing a probability distribution with the Maximum Likelihood Estimate
from nltk.probability import MLEProbDist

import operator
import random

############# INTRO POS #################

def intro():
  # NLTK provides corpora tagged with part-of-speech (POS) information and some tools to access this information
  # The Penn Treebank tagset is commonly used for English
  nltk.help.upenn_tagset()

  # We can retrieve the tagged sentences in the Brown corpus by calling the tagged_sents() function
  tagged_sentences = brown.tagged_sents(categories= 'news')
  print "Sentence tagged with Penn Treebank POS labels:"
  print tagged_sentences[42]

   # We can access the Universal tags by changing the tagset argument
  tagged_sentences_universal = brown.tagged_sents(categories= 'news', tagset='universal')
  print "Sentence tagged with Universal POS:"
  print tagged_sentences_universal[42]

# Comment to hide intro
intro()


############# EXERCISE 1 #################
# Solution for exercise 1
# Input: genre (string), tagset (string)
# Output: number_of_tags (int), top_tags (list of string)


# get the number of tags found in the corpus
# compute the Frequency Distribution of tags

def ex1(genre,tagset):
  
  # get the tagged words from the corpus
  tagged_words = brown.tagged_words(categories= genre, tagset=tagset)
  
  # TODO: build a list of the tags associated with each word
  #tags =
  
  # TODO: using the above list compute the Frequency Distribution of tags in the corpus
  # hint: use nltk.FreqDist()
  #tagsFDist =
  
  # TODO: retrieve the total number of tags in the tagset
  #number_of_tags =
  
  #TODO: retrieve the top 10 most frequent tags
  #top_tags =
  return (number_of_tags,top_tags)



def test_ex1():
  print "Tag FreqDist for news:"
  print ex1('news',None)

  print "Tag FreqDist for science_fiction:"
  print ex1('science_fiction',None)

  # Do the same thing for a different tagset: Universal

  print "Tag FreqDist for news with Universal tagset:"
  print ex1('news','universal')

  print "Tag FreqDist for science_fiction with Universal tagset:"
  print ex1('science_fiction','universal')

### Uncomment to test exerise 1
# Let's look at the top tags for different genre and tagsets
#  and observe the differences
#test_ex1()

############# EXERCISE 2 #################
# Solution for exercise 2
# Input: sentence (list of string), size (<4600)
# Output: hmm_tagged_sentence (list of tuples), tagger (HiddenMarkovModelTagger)

# hint: use the help on HiddenMarkovModelTagger to find out how to train, tag and evaluate the HMM tagger
def ex2(sentence, size):
  
  tagged_sentences = brown.tagged_sents(categories= 'news')
  
  # set up the training data
  train_data = tagged_sentences[-size:]
  
  # set up the test data
  test_data = tagged_sentences[:100]

  # TODO: train a HiddenMarkovModelTagger, using the train() method
  #tagger =

  # TODO: using the hmm tagger tag the sentence
  #hmm_tagged_sentence =
  
  # TODO: using the hmm tagger evaluate on the test data
  #eres =

  return (tagger, hmm_tagged_sentence,eres)


def test_ex2():
  tagged_sentences = brown.tagged_sents(categories= 'news')
  words = [tp[0] for tp in tagged_sentences[42]]
  (tagger, hmm_tagged_sentence, eres ) = ex2(words,500)
  print "Sentenced tagged with nltk.HiddenMarkovModelTagger:"
  print hmm_tagged_sentence
  print "Eval score:"
  print eres

  (tagger, hmm_tagged_sentence, eres ) = ex2(words,3000)
  print "Sentenced tagged with nltk.HiddenMarkovModelTagger:"
  print hmm_tagged_sentence
  print "Eval score:"
  print eres

### Uncomment to test exerise 2
#Look at the tagged sentence and the accuracy of the tagger. How does the size of the training set affect the accuracy?
#test_ex2()



############# EXERCISE 3 #################
# Solution for exercise 3
# Input: tagged_words (list of tuples)
# Output: emission_FD (ConditionalFreqDist), emission_PD (ConditionalProbDist), p_NN (float), p_DT (float)


# in the previous labs we've seen how to build a freq dist
# we need conditional distributions to estimate the transition and emission models
# in this exerise we estimate the emission model
def ex3(tagged_words):

  # TODO: prepare the data
  # the data object should be a list of tuples of conditions and observations
  # in our case the tuples will be of the form (tag,word) where words are lowercased
  #data =

  # TODO: compute a Conditional Frequency Distribution for words given their tags using our data
  #emission_FD =
  
  # TODO: return the top 10 most frequent words given the tag NN
  #top_NN =
  
  # TODO: Compute the Conditional Probability Distribution using the above Conditional Frequency Distribution. Use MLEProbDist estimator.
  #emission_PD =
  
  # TODO: compute the probabilities of P(year|NN) and P(year|DT)
  #p_NN =
  #p_DT =
  
  return (emission_FD, top_NN, emission_PD, p_NN, p_DT)


def test_ex3():
  tagged_words = brown.tagged_words(categories='news')
  (emission_FD, top_NN, emission_PD, p_NN, p_DT) = ex3(tagged_words)
  print "Frequency of words given the tag *NN*: ", top_NN
  print "P(year|NN) = ", p_NN
  print "P(year|DT) = ", p_DT

### Uncomment to test exerise 3
#Look at the estimated probabilities. Why is P(year|DT) = 0 ? What are the problems with having 0 probabilities and what can be done to avoid this?
#test_ex3()

############# EXERCISE 4 #################
# Solution for exercise 4
# Input: tagged_sentences (list)
# Output: emission_FD (ConditionalFreqDist), emission_PD (ConditionalProbDist), p_VBD_NN, p_DT_NN

# compute the transition probabilities
# the probabilties of a tag at position i+1 given the tag at position i
def ex4(tagged_sentences):
  
  # TODO: prepare the data
  # the data object should be an array of tuples of conditions and observations
  # in our case the tuples will be of the form (tag_(i),tag_(i+1))
  #data =
  

  # TODO: compute the Conditional Frequency Distribution for a tag given the previous tag
  #transition_FD =
  
  # TODO: compute the Conditional Probability Distribution for the
  # transition probability P(tag_(i+1)|tag_(i)) using the MLEProbDist
  # to estimate the probabilities
  #transition_PD =

  # TODO: compute the probabilities of P(NN|VBD) and P(NN|DT)
  #p_VBD_NN =
  #p_DT_NN =

  return (transition_FD, transition_PD,p_VBD_NN, p_DT_NN )


def test_ex4():
  tagged_sentences = brown.tagged_sents(categories= 'news')
  (transition_FD, transition_PD,p_VBD_NN, p_DT_NN ) = ex4(tagged_sentences)
  print "P(NN|VBD) = ", p_VBD_NN
  print "P(NN|DT) = ", p_DT_NN

### Uncomment to test exerise 4
# Are the results what you would expect? The sequence NN DT seems very probable. How will this affect the sequence tagging?
#test_ex4()
