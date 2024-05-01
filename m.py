import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # Importing load_model

# Load the necessary data
# You need to have the model, movie_df, and user2user_encoded, movie2movie_encoded dictionaries ready
model = load_model("movie_recommendation_model.h5")  # Load the model
ratings = pd.read_csv('ratings.csv')  # Load ratings.csv

movie_df = pd.read_csv("movies.csv")
user_ids = ratings["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = ratings["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
# adds encoded index columns to df
ratings["user"] = ratings["userId"].map(user2user_encoded)
ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)

# Function to get recommendations for a user
def get_recommendations(user_id, model, user2user_encoded, movie2movie_encoded, movie_df, ratings):
    movies_watched_by_user = ratings[ratings.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched_index = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched_index))

    ratings = model.predict([user_movie_array[:,0], user_movie_array[:,1]]).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched_index[x][0]) for x in top_ratings_indices]

    return recommended_movie_ids

# Streamlit app
def main():
    st.title("Movie Recommendation System")

    # Get user input
    user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=1000, step=1)

    # Display recommendations
    if st.button("Get Recommendations"):
        recommended_movie_ids = get_recommendations(user_id, model, user2user_encoded, movie2movie_encoded, movie_df, ratings)
        recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
        
        st.subheader("Top 10 Movie Recommendations")
        for index, row in recommended_movies.iterrows():
            st.write(f"{row['title']} - {row['genres']}")

if __name__ == "__main__":
    main()
