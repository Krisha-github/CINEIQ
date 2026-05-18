import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/cleaned_movies.csv")

df = df.head(100000)

user_movie_matrix = df.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)

user_movie_matrix = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


def recommend_movies_for_user(user_id, top_n=10):
    if user_id not in user_similarity_df.index:
        return f"User {user_id} not found."

    # Find users most similar to given user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Remove the user themself
    similar_users = similar_users.drop(user_id)

    # Take top 5 similar users
    top_users = similar_users.head(5).index

    # Movies already watched by current user
    watched_movies = user_movie_matrix.loc[user_id]
    watched_movies = watched_movies[watched_movies > 0].index

    # Ratings from similar users
    similar_users_ratings = user_movie_matrix.loc[top_users]

    # Average rating of movies by similar users
    movie_scores = similar_users_ratings.mean().sort_values(ascending=False)

    # Remove already watched movies
    recommendations = movie_scores.drop(watched_movies, errors="ignore")

    return recommendations.head(top_n)


print(recommend_movies_for_user(1, 10))