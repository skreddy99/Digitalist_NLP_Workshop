import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

f = open("training_text.txt","rb")

training_text_set = f.read()

s = open("sample.txt","rb")

sample_set = s.read()

my_tokenizer = PunktSentenceTokenizer(training_text_set)

my_tokens = my_tokenizer.tokenize(sample_set)

try:
    for i in my_tokens[:5]:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        print(tagged)

except Exception as e:
    print(str(e))
