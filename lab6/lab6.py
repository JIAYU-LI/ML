import operator

import nltk

from nltk.probability import FreqDist,MLEProbDist

from nltk.corpus import gutenberg
from nltk import bigrams
from nltk.corpus import stopwords
from math import log
from pprint import pprint


class FasterMLEProbDist(MLEProbDist):
  '''Speed up prob lookup for large sample sizes'''
  def __init__(self,freqdist):
    self._N=freqdist.N()
    if self._N == 0:
      self._empty = True
    else:
      self._empty = False
      self._pq=float(self._N)
    MLEProbDist.__init__(self,freqdist)

  def prob(self, sample):
    '''use cached quotient for division'''
    if self._empty:
      return 0
    else:
      return float(self._freqdist[sample]) / self._pq

sentences=gutenberg.sents('melville-moby_dick.txt')
stopwords = stopwords.words('english')



def Filter1(word):
  return word.isalpha()

def Filter2(word):
  return (word.isalpha() and not(word.lower() in stopwords))

#TODO
# Function for building the data set
# Input: a list of sentences, Filter function
# Build the list of bigrams and unigrams from the sentences and return this data

def BuildData(sentences,Filter):
  #unigrams_list = 
  #bigrams_list 
  return bigrams_list, unigrams_list



#TODO: using the data build the probability distribution over bigrams and unigrams using FasterMLEProbDist
def ex1(bigrams, unigrams):
  #TODO build the frequency distribution over bigrams and unigrams
  #bigramFreqDist = 
  #unigramFreqDist = 

  #TODO build the probability distribuition from the above frequency distributions using the FasterMLEProbDist estimator
  #bigramProbDist = 
  #unigramProbist = 

  return bigramProbDist, unigramProbist


def test1():
  bigrams, unigrams = BuildData(sentences,Filter1)
  
  bigramProbDist1, unigramProbist1 = ex1(bigrams, unigrams)
  print "type: ",type(bigramProbDist1) # <class 'nltk.probability.FasterMLEProbDist'>
  print "type: ",type(unigramProbist1) # <class 'nltk.probability.FasterMLEProbDist'>
  
  MLESorted = bigramProbDist1.freqdist().most_common(30)
  print "Using filter 1:",pprint(MLESorted)
  print "type: \n",type(MLESorted) # <type 'list'>

  bigrams, unigrams = BuildData(sentences,Filter2)
  bigramProbDist, unigramProbist = ex1(bigrams, unigrams)
  MLESorted = bigramProbDist.freqdist().most_common()[:30]
  print "Using filter 2:",pprint(MLESorted)
  print "\n"

  return bigramProbDist1, unigramProbist1
 

# TEST EXERCISE 1 - return values will be used for exercise 2
#bigramProbDist, unigramProbDist = test1()


#TODO: for each sample in the bigramProbDist compute the PMI and add {sample,PMI} pair to the PMI dict
#input: bigram and unigram distribution of type nltk.probability.FasterMLEProbDist
#output: PMI dict, PMIsorted list
def ComputePMI(bpd, upd):

  #PMIs = 
  #TODO: make a list of (sample,PMI) pairs for each sample in bpd

  #list of (bigrams,PMI) sorted according to the PMI score
  PMIsorted = sorted(PMIs, key=operator.itemgetter(1), reverse=True)

  return dict(PMIs), PMIsorted

def test2(bpd,upd):
  
  print "type: ",type(bpd) # <class 'nltk.probability.FasterMLEProbDist'>
  print "type: ",type(upd) # <class 'nltk.probability.FasterMLEProbDist'>
  
  PMIs, PMIsorted = ComputePMI(bpd, upd)
  print "type: ", type(PMIs) # <type 'dict'>
  print "type: ", type(PMIsorted) # <type 'list'>
  

  print "sperm whale %0.2f" % PMIs[("sperm","whale")]
  print "of the %0.2f" % PMIs[("of","the")]
  print "old man %0.2f" % PMIs[("old","man")] #comment why it's not as expected close to 0 -> because not enough data
  print "one side %0.2f" % PMIs[("one","side")]
  print  "\n"
  bcount=bpd.freqdist()
  for pair in  PMIsorted[:10]:
    print "%s\t%0.2f\t%d" % (pair[0], pair[1], bcount[pair[0]])
  
  n=0
  for pair in PMIsorted:
    if n==10:
      break
    if bcount[pair[0]]>30:
      print "%s\t%0.2f\t%d" % (pair[0], pair[1], bcount[pair[0]])
      n+=1
  print

  return PMIsorted

# TEST EXERCISE 2 - return values will be used for exercise 3
#PMIsorted = test2(bigramProbDist, unigramProbDist)

  
#TODO to eliminate the problem of low frequency put a threshold on the bigram frequency
def ex3(PMIsorted,bpd):

  #TODO we need a freqdist from which to pull bigram frequencies -- it's in bpd
  #bcount = 

  #TODO Return a list of bigrams and their corresponding PMI for bigrams composed of words with frequency greater than 30
  #high_freq_PMIsorted =
  return high_freq_PMIsorted

def test3(PMIsorted, bpd):

  high_freq = ex3(PMIsorted, bpd)

  print "\nTop 20 by PMI where pair count>30"
  print "%s\t%s\t%s"%('PMI','n','pair')
  bcount = bpd.freqdist()
  for pair in high_freq[:20]:
    print "%0.2f\t%d\t%s" % (pair[1], bcount[pair[0]], pair[0])

  print "\nBottom 20 by PMI where pair count>30"
  for pair in  high_freq[-20:]:
    print "%s\t%0.2f\t%d" % (pair[0], pair[1], bpd.freqdist()[pair[0]])
  
# TEST EXERCISE 3
#test3(PMIsorted,bigramProbDist)



