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


# # Define the model
class TfrsModel(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features[0]["user_idx"][:, None])
        movie_embeddings = self.movie_model(features[0]["movie_idx"][:, None])

        return self.task(user_embeddings, movie_embeddings)

    def call(self, features, training=False):
        user_embeddings = self.user_model(features["user_idx"][:, None])
        movie_embeddings = self.movie_model(features["movie_idx"][:, None])

        return self.task(user_embeddings, movie_embeddings)


# Create user and movie embedding layers using Embedding
user_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=len(user_mapping),
            output_dim=32,
            input_length=1,  # Add input_length
        ),
        # Add more layers as needed
    ]
)

movie_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(
            input_dim=len(movie_mapping),
            output_dim=32,
            input_length=1,  # Add input_length
        ),
        # Add more layers as needed
    ]
)


# Initialize and compile the model
tsfr_model = TfrsModel(user_model, movie_model)
tsfr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))


# Train the model
tsfr_model.fit(
    x={"user_idx": user_features, "movie_idx": movie_features}, y=ratings, epochs=5
)

# Define a sample input shape based on your actual data
sample_input_shape = {"user_idx": (None, 1), "movie_idx": (None, 1)}

# Build the model with the specified input shape
tsfr_model.build(input_shape=sample_input_shape)

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