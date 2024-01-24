import tensorflow as tf
import numpy as np

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