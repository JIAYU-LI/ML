import nltk
import sys

# Import the gutenberg corpus
from nltk.corpus import gutenberg

# Import NLTK's NgramModel module (for building language models)
# It has been removed from standard NLTK, so we access it in a special package installation
sys.path.extend(['/group/ltg/projects/fnlp', '/group/ltg/projects/fnlp/packages_2.6'])
from nltkx import NgramModel

# Import probability distributions
from nltk.probability import LaplaceProbDist
from nltk.probability import LidstoneProbDist
from nltk.probability import SimpleGoodTuringProbDist


#################### EXERCISE 0 ####################

# Solution for exercise 0
# Input: word (string), context (string)
# Output: p (float)
# Compute the unsmoothed (MLE) probability for word given the single word context
def ex0(word,context):
    p = 0.0

    austen_words = [w.lower() for w in gutenberg.words('austen-sense.txt')]
    austen_bigrams = zip(austen_words[:-1], austen_words[1:])  # list of bigrams as tuples
    # (above doesn't include begin/end of corpus: but basically this is fine)

    # Compute probability of word given context. Make sure you use float division.
    # p = ...

    # Return probability
    return p


# ### Uncomment to test exercise 0
# print "MLE:"
# result0a = ex0('end','the')
# print "Probability of \'end\' given \'the\': " + str(result0a)
# result0b = ex0('the','end')
# print "Probability of \'the\' given \'end\': " + str(result0b)


#################### EXERCISE 1 ####################

# Solution for exercise 1
# Input: word (string), context (string)
# Output: p (float)
# Compute the Laplace smoothed probability for word given the single word context
def ex1(word,context):
    p = 0.0

    austen_words = [w.lower() for w in gutenberg.words('austen-sense.txt')]
    austen_bigrams = zip(austen_words[:-1], austen_words[1:])  # list of bigrams as tuples
    # (above doesn't include begin/end of corpus: but basically this is fine)
    
    # compute the vocabulary size
    # V = ... 

    # Compute probability of word given context
    #p = ...

    # Return probability
    return p


### Uncomment to test exercise 1
# print "LAPLACE:"
# result1a = ex1('end','the')
# print "Probability of \'end\' given \'the\': " + str(result1a)
# result1b = ex1('the','end')
# print "Probability of \'the\' given \'end\': " + str(result1b)



#################### EXERCISE 2 ####################
# Solution for exercise 2
# Input: word (string), context (string), alpha (float)
# Output: p (float)
# Compute the Lidstone smoothed probability for word given the single word context
# Alpha is the smoothing parameter, normally between 0 and 1.
def ex2(word,context,alpha):
    p =0.0

    austen_words = [w.lower() for w in gutenberg.words('austen-sense.txt')]
    austen_bigrams = zip(austen_words[:-1], austen_words[1:])  # list of bigrams as tuples

    # compute the vocabulary size
    # V = ... 

    # Compute probability of word given context
    #p = ...

    # Return probability
    return p


### Uncomment to test exercise 2
# print "LIDSTONE, alpha=0.01:"
# result2a = ex2('end','the',.01)
# print "Probability of \'end\' given \'the\': " + str(result2a)
# result2b = ex2('the','end',.01)
# print "Probability of \'the\' given \'end\': " + str(result2b)
# print "LIDSTONE, alpha=0:"
# result2c = ex2('end','the',0)
# print "Probability of \'end\' given \'the\': " + str(result2c)
# result2d = ex2('the','end',0)
# print "Probability of \'the\' given \'end\': " + str(result2d)
# print "LIDSTONE, alpha=1:"
# result2e = ex2('end','the',1)
# print "Probability of \'end\' given \'the\': " + str(result2e)
# result2f = ex2('the','end',1)
# print "Probability of \'the\' given \'end\': " + str(result2f)



#################### EXERCISE 3 ####################
# Solution for exercise 3
# Input: word (string), context (string)
# Output: p (float)
def ex3(word,context):
    p =0.0

    austen_words = [w.lower() for w in gutenberg.words('austen-sense.txt')]

    # Train a bigram language model using a LAPLACE estimator AND BACKOFF
    # lm = NgramModel(2,austen_words,estimator=lambda f,b: LaplaceProbDist(f,b+1))
    # b+1 is necessary to provide, as it were, a bin for unseen contexts,
    # so there is some probability left for the backoff probability, i.e. so
    # that alpha is > 0.

    # Compute probability of word given context.
    # This method takes two arguments: the word and a *list* of words
    # specifying its context.
    # To see messages about backoff in action, use the named argument
    # verbose = True.
    # p = lm.prob(...)

    # Return probability
    return p


### Uncomment to test exercise 3
# print "BACKOFF WITH LAPLACE"
# result3a = ex3('end','the')
# print "Probability of \'end\' given \'the\': " + str(result3a)
# result3b = ex3('the','end')
# print "Probability of \'the\' given \'end\': " + str(result3b)


#################### EXERCISE 4 ####################

# Solution for exercise 4 - entropy calculation
# Input: lm (NgramModel language model), doc_name (string)
# Output: e (float)
def ex4_tot_entropy(lm,doc_name):
    e = 0.0

    # Construct a list of lowercase words from the document (test document)
    #doc_words = ...

   # Compute the TOTAL cross entropy of the text in doc_name
    #e = ...

    # Return the entropy
    return e

# Solution for exercise 4 - per-word entropy calculation
# Input: lm (NgramModel language model), doc_name (string)
# Output: e (float)
def ex4_perword_entropy(lm,doc_name):
    e = 0.0

    # Construct a list of lowercase words from the document (test document)
    # doc_words = ...

    # Compute the PER-WORD cross entropy of the text in doc_name
    # e = ...

    # Return the entropy
    return e


# Solution for exercise 4 - language model training
# Input: doc_name (string)
# Output: l (language model)
def ex4_lm(doc_name):
    l = None

    # Construct a list of lowercase words from the document (training data for lm)
    #doc_words = ...

    # Train a trigram language model using doc_words with backoff and a Lidstone probability distribution with +0.01 added to the sample count for each bin
    #l = NgramModel(<order>,<training_data>,estimator=lambda f,b:nltk.LidstoneProbDist(f,0.01,f.B()+1))

    # Return the language model
    return l

### Uncomment to test exercise 4
# lm4 = ex4_lm('austen-sense.txt')
# result4a = ex4_tot_entropy(lm4,'austen-emma.txt')
# print "Total cross-entropy for austen-emma.txt: " + str(result4a)
# result4b = ex4_tot_entropy(lm4,'chesterton-ball.txt')
# print "Total cross-entropy for chesterton-ball.txt: " + str(result4b)
# result4c = ex4_perword_entropy(lm4,'austen-emma.txt')
# print "Per-word cross-entropy for austen-emma.txt: " + str(result4c)
# result4d = ex4_perword_entropy(lm4,'chesterton-ball.txt')
# print "Per-word cross-entropy for chesterton-ball.txt: " + str(result4d)
