from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

stemmer = PorterStemmer()

sample_words = [????]


for words in sample_words:
    print(stemmer.stem(words))
