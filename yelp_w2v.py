from nltk import word_tokenize
from nltk import download
import os
download('punkt')


# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from gensim.models import word2vec

download('stopwords')  # Download stopwords list.

# Remove stopwords.
stop_words = stopwords.words('english')
 # Download data for tokenizer.

def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


import json
from smart_open import smart_open

# Business IDs of the restaurants.
ids = ['EAwh1OmG6t6p3nRaZOW_AA', 'pomGBqfbxcqPv14c3XH-ZQ','iCQpiavjjPzJ5_3gPD5Ebg', 'UBv8heCQR0RPnUQG0zkXIQ', '7m1Oa1VYV98UUuo_6i0EZg' ]

w2v_corpus = []  # Documents to train word2vec on (all 6 restaurants).
wmd_corpus = []  # Documents to run queries against (only one restaurant).
documents = []  # wmd_corpus, with no pre-processing (so we can see the original documents).

biz={}

with smart_open('yelp_dataset/yelp_academic_dataset_business.json', 'rb') as business_data_file:
    for line in business_data_file:
        json_line = json.loads(line)
        biz[json_line['business_id']]=json_line['name']

review_data_dict={}
votes={}

with smart_open('yelp_dataset/yelp_academic_dataset_review.json', 'rb') as data_file:
    for line in data_file:
        json_line = json.loads(line)

        if json_line['business_id'] not in ids:
            # Not one of the 6 restaurants.
            continue

        # Pre-process document.
        text = json_line['text']
        if json_line['business_id'] not in review_data_dict:
            review_data_dict[json_line['business_id']]=[text]
            votes[json_line['business_id']]=0
        review_data_dict[json_line['business_id']].append(text) # Extract text from JSON object.
        text = preprocess(text)

        # Add to corpus for training Word2Vec.
        w2v_corpus.append(text)
        #print(text)

        #if json_line['business_id'] == ids[0]:

            # Add to corpus for similarity queries.
        wmd_corpus.append(text)
        documents.append(json_line['text'])

# Train Word2Vec on all the restaurants.
model = word2vec.Word2Vec(w2v_corpus, workers=3, size=100,min_count=1)

# Initialize WmdSimilarity.
from pyemd import emd
from gensim.similarities import WmdSimilarity
num_best = 10
instance = WmdSimilarity(wmd_corpus, model, num_best=10)

while True:
    sent=input('Enter your query: ')
    #sent = 'The place is ideally located and provides great views. The food is great but the service is poor.'
    query = preprocess(sent)

    sims = instance[query]  # A query is simply a "look-up" in the similarity class.

    print ('Query:')
    print (sent)
    for i in range(num_best):
        print()
        #print ('sim = %.4f' % sims[i][1])
        for key in review_data_dict:
            rvws=review_data_dict[key]
            if documents[sims[i][0]] in rvws:
                votes[key]+=1
                #print(biz[key])

    for key, value in sorted(votes.items(), key=lambda k:k[1],reverse=True):
        print(value)
        print(biz[key])

        print('\t')
        print(review_data_dict[key][0])
        print('\t')
        print(review_data_dict[key][1])
        print('\t')
        print(review_data_dict[key][2])
        print('$$$$$$$$$$$$')
        #print (documents[sims[i][0]])
