NLTK

---
🔹 P1: Tokenization
``` python

!pip install nltk

import nltk
nltk.download('all')

from nltk.tokenize import sent_tokenize, word_tokenize

dataset = """Hello Mr. Watson, how are you doing today?
The weather is awesome.
The garden is green.
We should go out for a walk.
The sky is pinkish-blue.
You shouldn't eat cardboard."""


print(sent_tokenize(dataset))

for i in sent_tokenize(dataset):
    print(i)

print("word_tokenize", word_tokenize(dataset))


from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
print("TreebankWordTokenizer", tokenizer.tokenize(dataset))

text = """नमस्ते श्रीमान वाटसन,
आप आज कैसे हैं?
मौसम शानदार है।
बगीचा हरा है।
हम टहलने के लिए बाहर जाना चाहिए।
आसमान गुलाबी-नीला है।
आपको कार्डबोर्ड नहीं खाना चाहिए"""

print("Word_tokenize", word_tokenize(text))

print("Sentence_tokenize", sent_tokenize(text))

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
print("WordPunctTokenizer", tokenizer.tokenize(text))

tokenize = TreebankWordTokenizer()
print("TreebankWordTokenizer", tokenize.tokenize(text))
```
---
🔹 P2: Stopwords
``` python
!pip install nltk

import nltk
nltk.download('all')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

dataset = """English is a very important asset in the modern world.
It helps people communicate across countries.
Learning English can open many career opportunities.
Students should practice reading and writing every day."""

stop_words = set(stopwords.words('english'))
stop_words

print("Total count of stopwords:", len(stop_words))
words = word_tokenize(dataset)
print(words)

words = word_tokenize(dataset)
print(words)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

print(filtered_sentence)

print("After removing stopwords", len(filtered_sentence))

filtered_sentence1 = []

for w in words:
    if w in stop_words:
        filtered_sentence1.append(w)

print(filtered_sentence1)

print("Existing stopwords", len(filtered_sentence1))
```
---
🔹 P3: Stemming & Lemmatization
``` python
!pip install nltk
import nltk
nltk.download('all')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
words = ["program","programs","programer","programming","programmers"]

for w in words:
    print(w,":",ps.stem(w))

sentence = "Programmers program with programming languages"
words = word_tokenize(sentence)
for w in words:
    print(w,":",ps.stem(w))

from nltk.stem import PorterStemmer
e_word = ["wait","waiting","waited","waits"]
ps = PorterStemmer()
for w in e_word:
    rootWord = ps.stem(w)
    print(rootWord)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
sentence = "He was waiting for the boat at the dock, watching as the waves gently lapped against the pier"
words = word_tokenize(sentence)
ps = PorterStemmer()
for w in words:
    rootWord = ps.stem(w)
    print(w,rootWord)

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()
print("Porter Stemmer:-")
ps.stem("cats")
ps.stem("trouble")
ps.stem("troubling")
ps.stem("troubled")
print()

print("Lancaster Stemmer:-")
lancaster.stem("cats")
lancaster.stem("trouble")
lancaster.stem("troubling")
lancaster.stem("troubled")

word_list = ["friend", "friendship", "friends", "friendships", "companionship", "ally", "allies", "companions"]
print("{0:20}{1:20}{2:20}".format("Word","Porter Stemmer", "Lancaster Stemmer"))
for word in word_list:
  print("{0:20}{1:20}{2:20}".format(word, porter.stem(word), lancaster.stem(word)))

sentence = "Pythoners are very intelligent and work very pythonly and now they are pythoning their way to success."
porter.stem(sentence)

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Stemming for {} is {}".format(w, porter_stemmer.stem(w)))
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
  print("Lemma for {} is {}".format(w, wordnet_lemmatizer.lemmatize(w)))

```
---
🔹 P4: POS Tagging
``` python
!pip install nltk

import nltk
nltk.download('all')

from nltk.tokenize import wordpunct_tokenize
from nltk.tag import pos_tag

dataset = """The Taj Mahal, located in Agra, India, is one of the most iconic and beautiful monuments in the world"""

new_data = wordpunct_tokenize(dataset)
print(new_data)

pos_tag(new_data)

nltk.help.upenn_tagset()
```
---
🔹 P5: NER & Chunking
``` python
!pip install nltk

import nltk
nltk.download('all')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

dataset = """Abraham Lincoln was an American statesman, lawyer, and the 16th president of the United States"""

dataset_tag = pos_tag(word_tokenize(dataset))
print(dataset_tag)

data_ner = ne_chunk(dataset_tag)
print(data_ner)

# Chunking
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

dataset = """Taj Mahal is one of the world's most celebrated structures in the world. It is a stunning white marble mausoleum located in Agra, India."""

new_data = word_tokenize(dataset)
print(new_data)

postagging = pos_tag(new_data)
print(postagging)

sequence_chunk = """
chunk:
{<NNPS>+}
{<NNP>+}
{<NN>+}
"""

chunk = RegexpParser(sequence_chunk)
chunk_result = chunk.parse(postagging)
print(chunk_result)
```
---
🔹 P6: WordNet & Similarity
``` python
!pip install nltk

import nltk
nltk.download('all')
nltk.download('wordnet')

from nltk.corpus import wordnet
syns = wordnet.synsets("program")
print(syns)
print(syns[0])

print(syns[0].lemmas())
print(syns[0].lemmas()[0].name())

print(syns[0].definition())
print(syns[0].examples())

antonyms = []
def TofindAntonyms(x):
    for syn in wordnet.synsets(x):
        for lm in syn.lemmas():
            if lm.antonyms():
                antonyms.append(lm.antonyms()[0].name())
    return antonyms
print(set(TofindAntonyms("bright")))
print(set(TofindAntonyms("inactive")))
print(set(TofindAntonyms("good")))

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print("Synonyms: ",set(synonyms))
print("Antonyms: ",set(antonyms))

car = wordnet.synset('car.n.01')
automobile = wordnet.synset('automobile.n.01')
print("Similarity between car and automobile", car.path_similarity(automobile))

from nltk.corpus import wordnet as wn

w1 = wordnet.synset('run.v.01')
w2 = wordnet.synset('sprint.v.01')
print("Wu-Palmer Similarity between run and sprint", w1.wup_similarity(w2))

jump = wn.synset('jump.v.01')
leap = wn.synset('leap.v.01')
ship = wn.synset('ship.n.01')
print("Wu-Palmer Similarity between jump and leap", jump.wup_similarity(leap))

```
---
🔹 P7: WordCloud
``` python
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as npy
from PIL import Image

dataset = open("/content/Indian Motorcycle.txt", "r").read()
dataset = dataset.upper()

maskArray = npy.array(Image.open("/content/video-300x300.jpg"))

cloud = WordCloud(
    background_color="#000012",
    min_font_size=5,
    colormap="viridis",
    max_font_size=70,
    font_path='/content/LEMONMILK-Medium.otf',
    max_words=200,
    collocations=True,
    mask=maskArray,
    stopwords=set(STOPWORDS)
).generate(dataset)

# plot the WordCloud image
plt.figure(figsize=(6, 10), facecolor=None)
plt.imshow(cloud)
plt.axis("off")

```
---
🔹 P8: Text Summarization
``` python
!pip install nltk

import nltk
nltk.download('all')

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.tokenize import sent_tokenize, word_tokenize

dataset = open("/content/Indian Motorcycle.txt", encoding='cp1252').read()
dataset

def summarization(dataset):
    result = []
    
    for number, sentence in enumerate(sent_tokenize(dataset)):
        number_tokens = len(word_tokenize(sentence))
        
        tagged = nltk.pos_tag(word_tokenize(sentence))
        
        number_nouns = len([word for word, pos in tagged if pos in ['NN', 'NNP']])
        
        ners = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=False)
        
        number_ners = len([chunk for chunk in ners if hasattr(chunk, 'label')])
        
        score = (number_ners + number_nouns) / float(number_tokens)
        
        result.append((number, score, sentence))
    
    return result

summ = summarization(dataset)
summ

for i in sorted(summ, key=lambda x: x[1], reverse=True):
    i[2]

```
---
🔹 P9: Word2Vec
``` python
!pip install wikipedia
!pip install gensim

import nltk
nltk.download('all')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import string
import warnings
from gensim.models import Word2Vec
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import wikipedia
from wikipedia import search, page

titles = search("Data Science")
wikipage = page(titles[0])
wikipage.content
wikipage.categories
wikipedia.summary("Data Science", sentences=1)
wikipedia.search("Data Science")

def preprocessing(text):
    result = []
    sent = sent_tokenize(text)
    
    for sentence in sent:
        words = word_tokenize(sentence)
        tokens = [w for w in words if w.lower() not in string.punctuation]
        stopw = stopwords.words('english')
        tokens = [token for token in tokens if token.lower() not in stopw]+
        tokens = [word for word in tokens if len(word) >= 3]
        Lemma = WordNetLemmatizer()
        tokens = [Lemma.lemmatize(word) for word in tokens]
        result += [tokens]
    return result

text_p = preprocessing(wikipage.content)
text_p[0]

# Define your parameters
min_count = 2
vector_size = 50
window = 4
wikimodel = Word2Vec(text_p, min_count=min_count, vector_size=vector_size, window=window)

vocab = list(wikimodel.wv.index_to_key)
vocab[:20]

wikimodel.wv.most_similar(positive=["analysis", "field"], topn=3)
wikimodel.wv.similarity("data", "information")
wikimodel.wv.similarity("data", "science")
wikimodel.wv.similarity("data", "computing")

titles = search("statistic")
wikipage = page(titles[0])
wikipage.content
wikipage.categories
wikipedia.summary("statistic", sentences=1)
wikipedia.search("statistic")

def preprocessing(text):
    result = []
    sent = sent_tokenize(text)
    
    for sentence in sent:
        words = word_tokenize(sentence)
        
        tokens = [w for w in words if w.lower() not in string.punctuation]
        
        stopw = stopwords.words("english")
        tokens = [token for token in tokens if token not in stopw]
        
        tokens = [word for word in tokens if len(word) >= 3]
        
        lemma = WordNetLemmatizer()
        tokens = [lemma.lemmatize(word) for word in tokens]
        
        result += [tokens]
    
    return result

text_p = preprocessing(wikipage.content)

text_p[0]

# Define parameters
min_count = 2
vector_size = 50
window = 4

wikimodel = Word2Vec(text_p, min_count=min_count, vector_size=vector_size, window=window)

vocab = list(wikimodel.wv.index_to_key)

vocab[:20]

wikimodel.wv.most_similar(positive=["statistical", "hypothesis"], topn=3)

wikimodel.wv.similarity("probability", "sample")
wikimodel.wv.similarity("probability", "value")
wikimodel.wv.similarity("data", "error")

```
---
🔹 P10: Movie Review Classification
``` python
!pip install nltk

import nltk
nltk.download('all')

import random
from nltk.corpus import movie_reviews

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

random.shuffle(documents)

documents[1]

# Normalize the dataset
all_words = []
for w in movie_reviews.words():
  all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["love"])

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)
```
