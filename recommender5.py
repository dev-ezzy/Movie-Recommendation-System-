#importing modules
import pandas as pd
import numpy as np
from surprise import KNNBasic, accuracy
import streamlit as st
import joblib

# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)




def neural_recommender(user_id, n, svd_model, model):
    # Get movies not rated by the user
    to_recommend = set(movies['movieId'].unique()) - set(movies[movies['userId'] == user_id]['movieId'].unique())

    # Get SVD predictions for movies to recommend
    svd_preds = [svd_model.predict(user_id, movie_id).est for movie_id in to_recommend]

    # Create a DataFrame with SVD predictions
    df_with_svd = pd.DataFrame({'userId': [user_id] * len(to_recommend), 'movieId': list(to_recommend), 'svd_preds': svd_preds})

    # Use the neural network to predict ratings
    nn_preds = model.predict([df_with_svd['userId'], df_with_svd['movieId'], df_with_svd['svd_preds']])

    # Combine movie IDs with their predicted ratings
    recommendations = pd.DataFrame({'movieId': list(to_recommend), 'predicted_rating': nn_preds.flatten()})

    # Get the top N recommendations
    top_recommendations = recommendations.nlargest(n, 'predicted_rating')

    # Merge with movie information to get titles and genres
    top_recommendations = pd.merge(top_recommendations, modelling_data[['movieId', 'title', 'genres']], on='movieId', how='left')

    return top_recommendations[['movieId', 'title', 'genres', 'predicted_rating']]




def knn_caller(user_id, knn_model):
    user_id = user_id
    # Get a list of all movie IDs in your dataset
    all_movie_ids = np.unique(modelling_data['movieId'])

    # Create a list to store predicted ratings for unrated movies
    predicted_ratings = []

    # Predict ratings for the user on unrated movies
    for movie_id in all_movie_ids:
        # Check if the user has already rated the movie
        if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
            predicted_rating = knn_model.predict(user_id, movie_id)
            predicted_ratings.append((movie_id, predicted_rating.est))
            #Sort the predicted ratings in descending order
            predicted_ratings.sort(key=lambda x: x[1], reverse=True)

            # Get the top 5 movie recommendations
            top_5_recommendations = predicted_ratings[:5]

            # Display the top 5 recommended movies
            for movie_id, predicted_rating in top_5_recommendations:
                # Check if the condition results in a non-empty DataFrame
                if not movies[movies['movieId'] == movie_id].empty:
                    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
                    rounded_rating = round(predicted_rating, 1)
                    print(f"Movie: {movie_title}, Predicted Rating: {rounded_rating}")
                else:
                    print(f"Movie with ID {movie_id} not found in movies_df")
                    
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