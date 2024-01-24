import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import streamlit as st
import joblib
#data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
# merging the two dataframes
modelling_data = pd.merge(movies, ratings, on="movieId")
# dropping timestamp
modelling_data.drop(columns="timestamp", axis=1, inplace=True)

#Map user and movie IDs to unique integer indices
user_mapping = {id: idx for idx, id in enumerate(modelling_data["userId"].unique())}
movie_mapping = {id: idx for idx, id in enumerate(modelling_data["movieId"].unique())}

modelling_data["user_idx"] = modelling_data["userId"].map(user_mapping)
modelling_data["movie_idx"] = modelling_data["movieId"].map(movie_mapping)

# Create user and movie feature tensors
user_features = tf.convert_to_tensor(modelling_data["user_idx"], dtype=tf.int32)
movie_features = tf.convert_to_tensor(modelling_data["movie_idx"], dtype=tf.int32)
ratings = tf.convert_to_tensor(modelling_data["rating"], dtype=tf.float32)

def get_top_movie_predictions(user, user_mapping, movie_mapping, model, df):
    # Get the user index for the given user ID
    user_idx = user_mapping[user]

    # Create user feature tensor for the given user
    user_feature = tf.convert_to_tensor([user_idx], dtype=tf.int32)

    # Generate movie indices for all movies
    all_movie_indices = tf.range(len(movie_mapping), dtype=tf.int32)

    # Tile the user index to match the number of movies
    user_feature_tiled = tf.tile(user_feature, [len(movie_mapping)])

    # Create movie feature tensor for all movies
    movie_features_all = tf.convert_to_tensor(all_movie_indices, dtype=tf.int32)

    # Predict ratings for all movies
    predicted_ratings = model.predict(
        x={"user_idx": user_feature_tiled, "movie_idx": movie_features_all}
    )

    # Get the top 5 movie indices based on predicted ratings
    top5_movie_indices = np.argsort(predicted_ratings)[-5:][::-1]

    # Map movie indices back to movie IDs
    top5_movie_ids = [
        key for key, value in movie_mapping.items() if value in top5_movie_indices
    ]

    # Fetch and print details for the top 5 predicted movies
    for item in top5_movie_ids:
        movie_data = df[df["movieId"] == item]
        title = movie_data["title"].values[0]
        genres = movie_data["genres"].values[0]
        # Add any other columns you want to retrieve
        print(
            f"Recommendations for user {user} are: ....\n",
        )
        print(f"Movie Id: {item}, Title: {title}, Genres: {genres}")
        
        
# Streamlit UI
st.title("Ezra Recommendation System")

# # SVD Model
# st.subheader("SVD Recommendations")
# user_id_svd = st.number_input("Enter your User ID for SVD Recommendations", min_value=1, step=1)
# num_recommendations_svd = st.slider("Number of SVD Recommendations", 1, 10, 5)
# if st.button('Get SVD Recommendations'):
#     # Load your trained SVD model
#     svd_model = joblib.load("best_recommender.joblib")  # Corrected model path
#     recommended_movies_svd = recommender_system(user_id_svd, num_recommendations_svd, baseline_data, svd_model)
#     st.write(recommended_movies_svd)

# # KNN Model
# st.subheader("KNN Recommendations")
# user_id_knn = st.number_input("Enter your User ID for KNN Recommendations", min_value=1, step=1)
# num_recommendations_knn = st.slider("Number of KNN Recommendations", 1, 8, 5)
# if st.button('Get KNN Recommendations'):
#     # Load your trained KNN model
#     knn_model = joblib.load("best_recommender.joblib")  # Replace with your model path
#     recommended_movies_knn = knn_recommender(user_id_knn, num_recommendations_knn, knn_model)
#     st.write(recommended_movies_knn)

# TensorFlow Recommenders Model
st.subheader("TensorFlow Recommenders")
user_id_tf = st.number_input("Enter your User ID for TensorFlow Recommendations", min_value=1, step=1)
if st.button('Get TensorFlow Recommendations'):
    # Load your trained TensorFlow model
    tf_model = joblib.load("best_recommender.joblib")  # Replace with your model path
    recommended_movies_tf = get_top_movie_predictions(user_id_tf, user_mapping, movie_mapping, tf_model, modelling_data)
    st.write(recommended_movies_tf)                