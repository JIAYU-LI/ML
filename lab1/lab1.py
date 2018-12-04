import nltk
import sys

# Import the Presidential inaugural speeches corpus
from nltk.corpus import inaugural

# Import the gutenberg corpus                                                                               
from nltk.corpus import gutenberg

# Import NLTK's NgramModel module (for building language models)
# It has been removed from standard NLTK, so we access it in a special package installation
sys.path.extend(['/group/ltg/projects/fnlp', '/group/ltg/projects/fnlp/packages_2.6'])
from nltkx import NgramModel


#################### EXERCISE 1 ####################

# Solution for exercise 1
# Input: doc_name (string)
# Output: total_words (int), total_distinct_words (int)
def ex1(doc_name):
    # Use the plaintext corpus reader to access a pre-tokenised list of words
    # for the document specified in "doc_name"
    doc_words = inaugural.words(doc_name)

    # Find the total number of words in the speech
    #total_words =

    # Find the total number of DISTINCT words in the speech
    #total_distinct_words = 

    # Return the word counts
    return (total_words, total_distinct_words)


### Uncomment to test exercise 1
#speech_name = '2009-Obama.txt'
#(tokens,types) = ex1(speech_name)
#print "Total words in %s: %s"%(speech_name,tokens)
#print "Total distinct words in %s: %s"%(speech_name,types)



#################### EXERCISE 2 ####################

# Solution for exercise 2
# Input: doc_name (string)
# Output: avg_word_length (float)
def ex2(doc_name):
    doc_words = inaugural.words(doc_name)

    # Construct a list that contains the word lengths for each DISTINCT word in the document
    #distinct_word_lengths = 

    # Find the average word length
    #avg_word_length =

    # Return the average word length of the document
    return avg_word_length


### Uncomment to test exercise 2 
#speech_name = '2009-Obama.txt'
#result2 = ex2(speech_name)
#print "Average word length for %s: %s"%(speech_name,result2)


#################### EXERCISE 3 ####################

# Solution for exercise 3
# Input: doc_name (string), x (int)
# Output: top_words (list)
def ex3(doc_name, x):
    doc_words = inaugural.words(doc_name)
    
    # Construct a frequency distribution over the lowercased words in the document
    #fd_doc_words = 

    # Find the top x most frequently used words in the document
    #top_words = 

    # Return the top x most frequently used words
    return top_words


### Uncomment to test exercise 3
#print "Top 50 words for Obama's 2009 speech:"
#result3a = ex3('2009-Obama.txt', 50)
#print result3a
#print "Top 50 words for Washington's 1789 speech:"
#result3b = ex3('1789-Washington.txt', 50)
#print result3b



#################### EXERCISE 4 ####################

# Solution for exercise 4
# Input: doc_name (string), n (int)
# Output: lm (NgramModel language model)
def ex4(doc_name, n):
    # Construct a list of lowercase words from the document
    words = [w.lower() for w in gutenberg.words(doc_name)]

    # Build the language model using the nltk.MLEProbDist estimator 
    #lm = NgramModel(<order>,<training_data>)
    
    # Return the language model (we'll use it in exercise 5)
    return lm


### Uncomment to test exercise 4
#result4 = ex4('austen-sense.txt',2)
#print "Sense and Sensibility bigram language model built"



#################### EXERCISE 5 ####################

# Solution for exercise 5
# Input: lm (NgramModel language model, from exercise 4), word (string), context (list)
# Output: p (float)
def ex5(lm,word,context):
    # Compute the probability for the word given the context
    #p = 

    # Return the probability
    return p


### Uncomment to test exercise 5
#result5a = ex5(result4,'for',['reason'])
#print "Probability of \'reason\' followed by \'for\': %s"%result5a
#result5b = ex5(result4,'end',['the'])
#print "Probability of \'the\' followed by \'end\': %s"%result5b
#result5c = ex5(result4,'the',['end'])
#print "Probability of \'end\' followed by \'the\': %s"%result5c

### Uncomment to test exercise 6
#result6 = ex5(result4,'the',['end'],True)
