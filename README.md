# Yelp Recommendation

This is an experiment in which data science techniques are applied on the Yelp Dataset. First, we clean the data and filter it to obtain a particular subset of data. 
Then, we create 4 new features. These features aim to create collections in the restaurant data. 

We then train a word embedding model based on the reviews written by users for each restaurant. For a given natural query, we calculate document vectors, and find the reviews which are closest to the query. Our output/search results are the names of the restaurants with top matching reviews.
