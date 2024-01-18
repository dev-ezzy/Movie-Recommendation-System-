import pandas as pd
import numpy as np
from surprise import KNNBasic, accuracy


# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)



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
                    
                
                
# import numpy as np

# def knn_recommender(user_id, knn_model, N):
#     # Get a list of all movie IDs in your dataset
#     all_movie_ids = np.unique(modelling_data['movieId'])

#     # Create a list to store predicted ratings for unrated movies
#     predicted_ratings = []

#     # Predict ratings for the user on unrated movies
#     for movie_id in all_movie_ids:
#         #Check if the user has already rated the movie
#         if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
#             continue  # Skip if the user has rated the movie

#         predicted_rating = knn_model.predict(user_id, movie_id)
#         predicted_ratings.append((movie_id, predicted_rating.est))

#         # Sort the predicted ratings in descending order
#         predicted_ratings.sort(key=lambda x: x[1], reverse=True)

#         #Get the top 5 movie recommendations
#         top_5_recommendations = predicted_ratings[:N]

#         for movie_id, predicted_rating in top_5_recommendations:
#             # Check if the condition results in a non-empty DataFrame
#             if not movies[movies['movieId'] == movie_id].empty:
#                 movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
#                 rounded_rating = round(predicted_rating, 1)
#                 print(f"Movie: {movie_title}, Predicted Rating: {rounded_rating}")
#             else:
#                 print(f"Movie with ID {movie_id} not found in movies_df")


def knn_recommender(user_id, knn_model, N):
    # Get a list of all movie IDs in your dataset
    all_movie_ids = np.unique(movies['movieId'])

    # Check if the user has already rated any movies
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    # Filter unrated movies
    unrated_movies = np.setdiff1d(all_movie_ids, user_rated_movies)

    # Create a list to store predicted ratings for unrated movies
    predicted_ratings = []

    # Predict ratings for the user on unrated movies
    for movie_id in unrated_movies:
        predicted_rating = knn_model.predict(user_id, movie_id)
        predicted_ratings.append((movie_id, predicted_rating.est))

    # Sort the predicted ratings in descending order
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get the top N movie recommendations
    top_n_recommendations = predicted_ratings[:N]

    # Display the top N recommended movies
    for movie_id, predicted_rating in top_n_recommendations:
        # Check if the condition results in a non-empty DataFrame
        if not movies[movies['movieId'] == movie_id].empty:
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            rounded_rating = round(predicted_rating, 1)
            print(f"Movie: {movie_title}, Predicted Rating: {rounded_rating}")
        else:
            print(f"Movie with ID {movie_id} not found in movies_df")
