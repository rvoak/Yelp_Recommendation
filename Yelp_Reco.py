
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
import pandas as pd
from glob import glob
import nltk
from nltk.corpus import stopwords
import string

import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel

import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

df_business = pd.read_csv("yelp_business.csv")
df_review = pd.read_csv("yelp_review.csv")
df_business.drop("neighborhood", axis = 1, inplace =True)
df_business.dropna(inplace = True)
df_review=df_review.rename(columns = {'stars':'review_rating'})
df_review.dropna(inplace = True)
temp_df=df_review.groupby("business_id")["review_id"].count().reset_index()
temp_df=temp_df[temp_df['review_id']>70]
temp_df.set_index('business_id')

get_ipython().run_line_magic('matplotlib', 'inline')
axes = sns.distplot(temp_df['review_id'])
axes.set_xlim([0,500])
plt.xticks([50,100,150,200,250,300,350,400,450,500])

df_review=df_review[df_review['business_id'].isin(temp_df['business_id'])]
df_business["categories"]=df_business["categories"].str.split(";")
df_new=df_business.copy()

def is_restaurant(cat):
    return ('Restaurants' in cat)

df_business['is_restaurant']=df_business['categories'].apply(is_restaurant)
df_business=df_business[df_business['is_restaurant']]
df_business.drop('is_restaurant',axis=1,inplace=True)
merged_df = pd.merge(df_business, df_review, left_on = ["business_id"], right_on = ["business_id"], how = "inner")

def has_offers(review):
    to_search=['offers','discount','deals']
    for word in to_search:
        if(word in review):
            return True
    return False

merged_df['OFFERS-DISCOUNTS']=merged_df['text'].apply(has_offers)

def is_date_spot(review):
    to_search=['romantic','couple','date','sex','tinder']
    for word in to_search:
        if(word in review):
            return True
    return False

merged_df['DATE SPOT']=merged_df['text'].apply(is_date_spot)

def is_sports(review):
    to_search=['sports','baseball','rugby','game','football']
    for word in to_search:
        if(word in review):
            return True
    return False

merged_df['SPORTS BAR']=merged_df['text'].apply(is_sports)

def is_waffle(review):
    to_search=['waffle']
    for word in to_search:
        if(word in review):
            return True
    return False

merged_df['WAFFLES']=merged_df['text'].apply(is_waffle)

yelp_class = merged_df[(merged_df['state'] == "OH") ]

def text_process(text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

X = yelp_class['text']
y = yelp_class['review_rating']

bow_transformer = CountVectorizer(analyzer=text_process).fit(X)

print(len(bow_transformer.vocabulary_))

X = bow_transformer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)

filename = 'finalized_model.sav'
pickle.dump(bow_transformer, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

def Multinomial_Naive_Bayes():
    nb = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(y_test, preds))
    print('\n')
    print(classification_report(y_test, preds))

new_merged_df = merged_df.reset_index(drop=True)
z = merged_df["text"]
new_df = z.reset_index(drop=True)

def tokens(x):
    return x.split(',')

def get_vectorizer(train):
    vectorizer = TfidfVectorizer(stop_words ='english')
    X = vectorizer.fit_transform(train)
    return X, vectorizer

tfidf, vectorizer = get_vectorizer(new_df)
name = [input("Enter Query: ")]
tfidf_name = vectorizer.transform(name)

#Calculating the cosine distance between the review entered by the user and the all the reviews in the dataset

cosine_similarities = linear_kernel(tfidf_name, tfidf).flatten()
related_reviews_indices = cosine_similarities.argsort()[:-10:-1]
cosine_similarities[related_reviews_indices]

#Displaying the final restaurants that had similar reviews
new_merged_df["name"][related_reviews_indices]



#
##### Use this code for iPython Notebook only ####
#
'''
import ipywidgets as widgets
from ipywidgets import interact, interactive
from IPython.display import display
from ipywidgets import link


def on_search(b):
    global output
    global query_entry
    
    output.clear_output()

    name = [query_entry.value]
    tfidf_name = vectorizer.transform(name)

    cosine_similarities = linear_kernel(tfidf_name, tfidf).flatten()
    related_reviews_indices = cosine_similarities.argsort()[:-10:-1]
    #cosine_similarities[related_reviews_indices]

    #for indices in related_reviews_indices:
    with output:

            print(new_merged_df.loc[list(related_reviews_indices),['name','city','text']])

            print("\n")


def on_datespot(b):
    global output
    global merged_df

    output.clear_output()

    with output:
        for each in merged_df[merged_df['DATE SPOT']==True]['name'].unique():
            print('-----',each)

def on_sportsbar(b):
    global output
    global merged_df

    output.clear_output()

    with output:
        for each in merged_df[merged_df['SPORTS BAR']==True]['name'].unique():
            print('-----',each)

def on_waffle(b):
    global output
    global merged_df

    output.clear_output()

    with output:
        for each in merged_df[merged_df['WAFFLES']==True]['name'].unique():
            print('-----',each)


def on_offer(b):
    global output
    global merged_df

    output.clear_output()

    with output:
        for each in merged_df[merged_df['OFFERS-DISCOUNTS']==True]['name'].unique():
            print('-----',each)



query_entry=widgets.Textarea(
    value='',
    placeholder='Type something',
    description='QUERY:',
    disabled=False
)



btn_datespot=widgets.Button(
    description='DATE SPOT',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon=''
)

btn_sportsbar=widgets.Button(
    description='SPORTS BAR',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon=''
)

btn_waffle=widgets.Button(
    description='WAFFLE',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon=''
)

btn_offers=widgets.Button(
    description='OFFERS/DISCOUNTS',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon=''
)










btn_search=widgets.Button(
    description='Search',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon=''
)

output = widgets.Output(layout={'border': '1px solid black'})


display(widgets.HBox((btn_datespot,btn_sportsbar,btn_waffle,btn_offers,query_entry,btn_search)))
print('\n\n')
display(output)
type(btn_search)

btn_search.on_click(on_search)
btn_datespot.on_click(on_datespot)
btn_sportsbar.on_click(on_sportsbar)
btn_waffle.on_click(on_waffle)
btn_offers.on_click(on_offer)
'''
