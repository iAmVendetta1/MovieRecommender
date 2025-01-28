#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from fuzzywuzzy import fuzz 
from fuzzywuzzy import process


# In[2]:


# Load datasets
movies = pd.read_csv('movies.csv', sep=',')
ratings = pd.read_csv('ratings.csv', sep=',')

# Merge movies and ratings dataframes
data = pd.merge(ratings, movies, on='movieId')

# Check for missing values
#print(movies.isnull().sum())
#print(ratings.isnull().sum())


# In[3]:


# Extract features from the movies' genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate cosine similarity between movies
movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_similarity_df = pd.DataFrame(movie_similarity, index=movies['title'], columns=movies['title'])

# Display similarity matrix
movie_similarity_df.head()


# In[4]:


# Calculate average ratings for each movie
average_ratings = data.groupby('title')['rating'].mean()
average_ratings_df = pd.DataFrame(average_ratings)

# Display average ratings
average_ratings_df.head()


# In[19]:


# Function to get movie recommendations based on a given movie
def get_movie_recommendations(user_input, num_recommendations):
    # Find the best match for the user input
    best_match = process.extractOne(user_input, movie_similarity_df.columns, scorer=fuzz.partial_ratio)
    
    if best_match is None or best_match[1] < 60:  # Use a threshold to filter poor matches
        return "Movie not found."

    movie_title = best_match[0]
    
    # Get similar movies
    similar_movies = movie_similarity_df[movie_title].sort_values(ascending=False).index[1:]
    similar_scores = movie_similarity_df[movie_title].sort_values(ascending=False).values[1:]

    # Combine similarity scores with average ratings
    recommendations = []
    for movie, score in zip(similar_movies, similar_scores):
        if movie in average_ratings_df.index:
            avg_rating = average_ratings_df.loc[movie, 'rating']
            recommendations.append((movie, score, avg_rating))

    # Sort recommendations by a combined score (e.g., similarity + average rating)
    recommendations.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Return top N recommendations
    return movie_title, recommendations[:num_recommendations]

# Get user input for the movie title
user_movie = input("Enter the movie title you like: ")

def get_integer_input(prompt):
    while True:
        try:
            # Attempt to convert the input to an integer
            user_input = int(input(prompt))
            return user_input
        except ValueError:
            # If input is not a valid integer, prompt the user again
            print("Invalid input. Please use only numbers in this field.")

num_recommendations = get_integer_input("How many recommendations would you like? ")
result = get_movie_recommendations(user_movie, num_recommendations)

# Display recommendations
if isinstance(result, str):  # Check if the result is an error message
    print(result)
else:
    movie_title, recommendations = result
    print()
    print(f"Recommended Movies for '{movie_title}':")
    print()
    for movie, score, rating in recommendations:
        print(f"{movie:<70} Similarity: {score:<10.2f} Rating: {rating:.2f}")

# Provides a pause and option to exit when ready
input("Press Enter to exit...")

