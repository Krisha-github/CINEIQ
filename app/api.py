from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath("src"))

from hybrid_recommender import hybrid_recommend
from content_recommender import recommend_similar_movies
from sentiment import rerank_with_sentiment

app = FastAPI(
    title="CINEIQ Movie Recommendation API",
    description="Hybrid movie recommendation engine using content filtering, collaborative filtering, and SVD.",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "Welcome to CINEIQ API"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):
    results = hybrid_recommend(user_id=user_id, top_n=top_n)
    results = rerank_with_sentiment(results)
    return {
        "user_id": user_id,
        "recommendations": results
    }


@app.get("/similar")
def similar(movie_title: str, top_n: int = 10):
    results = recommend_similar_movies(movie_title, top_n)

    if isinstance(results, str):
        return {"error": results}

    return {
        "movie": movie_title,
        "similar_movies": results.to_dict(orient="records")
    }