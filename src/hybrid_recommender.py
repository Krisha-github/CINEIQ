import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment import get_sentiment_score
from explainability import generate_explanation

df = pd.read_csv("data/processed/cleaned_movies.csv")
df = df.head(100000)

movies = df[["movieId", "title", "genres", "metadata", "cast", "director", "keywords"]].drop_duplicates().reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(movies["metadata"].fillna(""))
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(range(len(movies)), index=movies["movieId"]).drop_duplicates()

user_movie_matrix = df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
svd_model = SVD()
svd_model.fit(trainset)


def hybrid_recommend(user_id, top_n=10):
    if user_id not in user_movie_matrix.index:
        return f"User {user_id} not found."

    watched_movies = df[df["userId"] == user_id]["movieId"].unique()
    candidate_movies = movies[~movies["movieId"].isin(watched_movies)]

    user_ratings = df[df["userId"] == user_id]
    liked_movies = user_ratings[user_ratings["rating"] >= 4.0]["movieId"].unique()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_similar_users = similar_users.head(5).index
    similar_users_ratings = df[df["userId"].isin(top_similar_users)]

    recommendations = []

    for _, movie in candidate_movies.iterrows():
        movie_id = movie["movieId"]

        svd_score = svd_model.predict(user_id, movie_id).est / 5.0

        content_score = 0
        if len(liked_movies) > 0 and movie_id in movie_index:
            target_idx = movie_index[movie_id]
            scores = []
            for liked_movie in liked_movies:
                if liked_movie in movie_index:
                    liked_idx = movie_index[liked_movie]
                    scores.append(content_similarity[target_idx][liked_idx])
            if scores:
                content_score = sum(scores) / len(scores)

        similar_movie_ratings = similar_users_ratings[
            similar_users_ratings["movieId"] == movie_id
        ]
        if len(similar_movie_ratings) > 0:
            collaborative_score = similar_movie_ratings["rating"].mean() / 5.0
        else:
            collaborative_score = 0

        proxy_text = movie["title"] + " " + movie["genres"]
        sentiment_score = get_sentiment_score(proxy_text)

        final_score = (
            0.4 * svd_score
            + 0.25 * content_score
            + 0.15 * collaborative_score
            + 0.2 * sentiment_score
        )

        explanation = generate_explanation({
            "content_score": content_score,
            "collaborative_score": collaborative_score,
            "svd_score": svd_score,
            "sentiment_score": sentiment_score,
            "director": movie["director"],
            "cast": movie["cast"]
        })

        recommendations.append({
            "movieId": int(movie_id),
            "title": movie["title"],
            "genres": movie["genres"],
            "cast": movie["cast"],
            "director": movie["director"],
            "keywords": movie["keywords"],
            "svd_score": round(svd_score, 3),
            "content_score": round(content_score, 3),
            "collaborative_score": round(collaborative_score, 3),
            "sentiment_score": round(sentiment_score, 3),
            "explanation": explanation,
            "final_score": round(final_score, 3)
        })

    return sorted(recommendations, key=lambda x: x["final_score"], reverse=True)[:top_n]


if __name__ == "__main__":
    results = hybrid_recommend(1, 10)
    for r in results:
        print(r["title"], "|", r["final_score"], "|", r["explanation"])