import streamlit as st
import pickle
import pandas as pd
import surprise as SVD

# Load data back from the file
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

st.title ("Movie Recommendation")
st.write("This is a movie recommendation system")

user_id = st.number_input("Enter User ID", min_value=1, max_value=610)

if st.button("Get Recommendation"):
  rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
  unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
  pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
  sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
  top_recommendations = sorted_predictions[:10]
  st.write(f"\nTop 10 movie recommendations for User {user_id}:")
  for recommendation in top_recommendations:
      movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
      st.write(f"{movie_title} (Estimated Rating: {recommendation.est})")

