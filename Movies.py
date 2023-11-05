import numpy as np
import pandas as pd
import difflib#user has to give the favourite movie name and there is achance the user might enter some 
#wrong spelling and this library will help in getting the closest name on the data set
from sklearn.feature_extraction.text import TfidfVectorizer
#this will convert the text from the dtatset into meaningful vectors
from sklearn.metrics.pairwise import cosine_similarity
#provide similarity score for particular  movies as compared to all the movies
# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv('movies.csv')
# printing the first 5 rows of the dataframe
print(movies_data.head())
# number of rows and columns in the data frame

movies_data.shape
# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)
# replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
  # combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)
# converting the text data to feature vectors

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)
# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)
print(similarity)
# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')
# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)
close_match = find_close_match[0]
print(close_match)
# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)
# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)
len(similarity_score)
# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)
# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Load the movie data from the CSV file into a pandas DataFrame
movies_data = pd.read_csv('movies.csv')

# Function to get movie recommendations based on user input
def get_movie_recommendations():
    movie_name = entry.get()

    # Find the closest match to the user's input in the dataset
    find_close_match = difflib.get_close_matches(movie_name, movies_data['title'].tolist(), n=1)
    
    if not find_close_match:
        messagebox.showinfo("Movie Recommendation", "No close match found for the given movie name.")
        return
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data['title'] == close_match]['index'].values[0]

    # Get similarity scores for the selected movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Display the recommended movies
    recommendations = "Movies suggested for you:\n"
    for i, movie in enumerate(sorted_similar_movies[:30], start=1):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommendations += f"{i}. {title_from_index}\n"

    messagebox.showinfo("Movie Recommendation", recommendations)

# Preprocess and calculate similarity
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# Create the Tkinter GUI
root = tk.Tk()
root.title("Movie Recommendation")

# Style the GUI
root.configure(bg="#F2F2F2")
root.geometry("400x200")

label = tk.Label(root, text="Enter your favourite movie name:", font=("Arial", 14), bg="#F2F2F2")
label.pack(pady=10)

entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=10)

recommend_button = tk.Button(root, text="Get Recommendations", font=("Arial", 12), command=get_movie_recommendations)
recommend_button.pack(pady=10)

root.mainloop()