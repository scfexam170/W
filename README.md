# implement Link Analysis
```

!pip install requests beautifulsoup4

from urllib.parse import urlparse
url= "https://www.google.com/search?q=pookie+meaning&rlz=1C1FKPE_enIN1104IN1104&oq=POOKIE&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgAQyBggBEEUYOTIHCAIQLhiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIPCAYQABgKGIMBGLEDGIAEMgYIBxAFGEDSAQc5NTdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8"
parsed_url = urlparse(url)
print("Parsed URL Components:")
print("Scheme: ",parsed_url.scheme)
print("Netlocation: ",parsed_url.netloc)
print("Path: ",parsed_url.path)
print("Query: ",parsed_url.query)
```
#program to make a web graph
```
import networkx as nx
import matplotlib.pyplot as plt
# create a directed graph
G = nx.DiGraph()
# Add edges representing links between web pages
edges = [
    ("Home","About"),
    ("Home", "Services"),
    ("About","Team"),
    ("Services","Contact"),
    ("Contact","Home"),
    ("Services","About")
]
G.add_edges_from(edges)
#Draw the graph
pos = nx.spring_layout(G) #positions from all nodes
nx.draw(G,pos,with_labels = True, arrows=True)
plt.title("Website Link Graph")
plt.show()
```
#Pr3 simple crawling from Wikipedia using keywords

```
import requests
from bs4 import BeautifulSoup
def simple_crawler(keyword):
    url = f"https://en.wikipedia.org/wiki/Special:Search?search={keyword}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    for link in soup.find_all('a',href=True):
        if link['href'].startswith('/wiki/') and ':' not in link['href']:
            print(f"https://en.wikipedia.org{link['href']}")
keyword = input("Enter a keyword to search: ")
simple_crawler(keyword)
```
#Pr4  fetch page content
```
import requests
def fetch_page(url):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses
        return response.text
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
url=" https://en.wikipedia.org/wiki/YouTube" # Replace with the target URL
page_content = fetch_page(url)
if page_content:
    print(page_content)
```
#Pr5 Hits Algorithm
```
import networkx as nx
G = nx.DiGraph()
G.add_edges_from([(1,2), (1,3), (2,4), (3,4), (4,4)])
# G.add_edges_from([('p','q'), ('p','s'), ('p','r'), ('q','r'), ('r','p'), ('s','s')])
authority_score, hub_score = nx.hits(G)
print(f"Authority Scores: {authority_score}")
print(f"Hub Scores: {hub_score}")
```
#Pr6 Page Rank
```
# Page rank using NetworkX
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_edges_from([
    ('A','B'),('A','C'),('B','A'),('B','D'),('B','E'),('B','F'),
    ('C','A'),('C','F')])
pr = nx.pagerank(G,alpha=0.85)
for node, rank in pr.items():
    print(f"Node -> {node}:{rank}")
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos,node_size=[pr[node]*1000 for node in G.nodes()])
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
plt.show()
```
#Pr7  summarize Text 
```
!pip install sumy

import nltk
nltk.download('punkt_tab')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_text(text, num_sentences=3):
    # Initialize the parser with the provided text
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Initialize the LexRankSummarizer
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summarized_text = ""
    for sentence in summary:
        summarized_text += str(sentence) + " "
    return summarized_text
text = '''The narrator explains that Kamban's poem begins by describing the land of Kosala,
    where the story takes place.He describes the people at work on the land, and the
    animals that live among them. The capital city, Ayodhya, is a fabulous city ruled
over by King Dasaratha. Though Dasaratha is a compassionate and well-loved king, he
laments that he's childless. One day, he mentions to his mentor that he has no sons
to succeed him,
    and asks his mentor for help.'''
summary = summarize_text(text)
print(summary)
```
#Prct 8 implement a Recommender System.
```
!pip uninstall -y numpy
!pip install numpy==1.23.5
!pip install scikit-surprise --no-binary :all:

!pip install pandas scikit-surprise

import pandas as pd
from surprise import Dataset,Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split

data = {
    'user_id': [1,1,1,2,2,3,3,3,4,4],
    'item_id': ['A','B','C','A','C','A','B','D','B','C'],
    'rating' : [5,3,4,4,8,6,1,2,4,1]
}

df = pd.DataFrame(data)
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(df[['user_id','item_id','rating']],reader)
trainset, testset = train_test_split(dataset, test_size=0.2) # test_size=0.2 means testing only 20% data
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")
pred = model.predict(3,'B')
print(f"Prediction for user 1, item A: {pred.est}")

```
