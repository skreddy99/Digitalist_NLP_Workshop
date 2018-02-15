'''

Sentiment analyses of pos.txt and neg.txt file with
a collection of positive comments and negative comments
This file pre-processes the data. Here is the brief algorithm
for pre-processing:

1. Get the statements sorted as positive and negative
2. Store list of positive and negative statements in separate files
and call them pos.txt and neg.txt
3. Next create a lexicon of the words in pos and neg statements.
To arrive at the lexicon, first tokenize the words in each file.
Later convert the words into lower case. Next eliminate those
words that occur too frequently and too rarely. The logic is
that in either case the impact of these words on the sentiment is less
4. Now convert each sentence of pos file into a vector of 1s and 0s
using the following logic: First create an empty sentence array of 0s, of
length same as the lexicon. Now read each sentence of the positive
file. For each word in the sentence, find if there is a
matching word in the lexicon. If yes, replace the 0 in the empty sentence array
with a 1 at the same index value as the matching word in lexicon.
At the end of processing of each sentence, the sentence array
will be a collection of 0s and 1s. Each sentence array is of the same
length, same as the lexicon and will be filled with 1s and 0s. This is
called sentence vector.
5. Append the classification identifier to
each sentence vector. The classification identifier is an array
[1,0] for a positive statement and [0,1] for a negative
statement. For example, a sentence vector for a positive sentence
will look like [....1,0,0,0,0,0,1....][1,0].
6. Repeat the same process for the neg file. The example of
a negative sentence vector will look like [....0,0,0,0,1,1,1....][0,1]
7. Merge these two list of sentence vectors into a single file and shuffle them.
8. Later divide the data set into training set and test set (10%).


'''
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemm = WordNetLemmatizer()

def create_lexicon(pos,neg):
    lexicon= []
    with open(pos,'r') as f:
        contents = f.readlines()
        for l in contents[:]:
            all_words = word_tokenize(l.lower())
            lexicon+= list(all_words)

    with open(neg,'r') as f:
        contents = f.readlines()
        for l in contents[:]:
            all_words = word_tokenize(l.lower())
            lexicon+= list(all_words)

    lexicon = [lemm.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 =[]
    for w in w_counts:
        if 1000 > w_counts[w] > 5:
            l2.append(w)
    print('\n\n\n' + 'The length of lexicon is ')
    print(len(l2))
    print('\n\n\n')
    return l2


def sample_handling(sample,lexicon,classification):
    featureset = []

    with open (sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:]:
            current_words = word_tokenize(l.lower())
            current_words = [lemm.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []

    features += sample_handling('pos.txt', lexicon,[1,0])
    features += sample_handling('neg.txt',lexicon,[0,1])

    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y

train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
with open ('sentiment_set.pickle', 'wb') as f:
    pickle.dump([train_x,train_y,test_x,test_y],f)


