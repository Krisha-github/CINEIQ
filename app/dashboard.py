import sys
import os
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.append(os.path.abspath("src"))

from hybrid_recommender import hybrid_recommend
from sentiment import rerank_with_sentiment

df = pd.read_csv("data/processed/cleaned_movies.csv")

st.set_page_config(page_title="CINEIQ Dashboard", layout="wide")
st.title("CINEIQ Movie Recommendation Dashboard")

user_id = st.number_input("Enter User ID", min_value=1, value=1, step=1)
user_data = df[df["userId"] == user_id].copy()

if user_data.empty:
    st.error("User not found.")
else:
    st.subheader(f"User {user_id} Taste Profile")

    c1, c2, c3 = st.columns(3)
    c1.metric("Movies Rated", len(user_data))
    c2.metric("Average Rating", round(user_data["rating"].mean(), 2))
    c3.metric("Highest Rating", user_data["rating"].max())

    st.subheader("Genre Preferences")
    genre_data = user_data.copy()
    genre_data["genres"] = genre_data["genres"].str.split("|")
    genre_data = genre_data.explode("genres")
    genre_scores = (
        genre_data.groupby("genres")["rating"]
        .mean()
        .reset_index()
        .sort_values("rating", ascending=False)
    )
    fig1 = px.line_polar(
        genre_scores,
        r="rating",
        theta="genres",
        line_close=True,
        title="Genre Radar Chart"
    )
    fig1.update_traces(fill="toself")
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Decade Preferences")
    user_data["year"] = user_data["title"].str.extract(r"\((\d{4})\)")
    user_data["year"] = pd.to_numeric(user_data["year"], errors="coerce")
    user_data["decade"] = (user_data["year"] // 10) * 10
    decade_scores = (
        user_data.groupby("decade")["rating"]
        .mean()
        .reset_index()
        .dropna()
    )
    fig2 = px.line(
        decade_scores,
        x="decade",
        y="rating",
        markers=True,
        title="Average Rating by Decade"
    )
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Director Affinity")
    director_data = user_data[user_data["director"].notna() & (user_data["director"] != "")]
    director_scores = (
        director_data.groupby("director")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["mean", "count"], ascending=False)
        .head(10)
    )
    fig3 = px.bar(
        director_scores,
        x="director",
        y="mean",
        title="Top Directors by Average Rating"
    )
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Actor Affinity")
    actor_data = user_data.copy()
    actor_data["cast"] = actor_data["cast"].fillna("").str.split(" ")
    actor_data = actor_data.explode("cast")
    actor_data = actor_data[actor_data["cast"] != ""]
    actor_scores = (
        actor_data.groupby("cast")["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["mean", "count"], ascending=False)
        .head(10)
    )
    fig4 = px.bar(
        actor_scores,
        x="cast",
        y="mean",
        title="Top Actors by Average Rating"
    )
    st.plotly_chart(fig4, width="stretch")

    st.subheader("Hybrid Recommendations")
    if st.button("Generate Recommendations"):
        recommendations = hybrid_recommend(user_id=user_id, top_n=10)
        recommendations = rerank_with_sentiment(recommendations)
        for rec in recommendations:
            st.markdown(f"**{rec['title']}** — Score: `{rec['final_sentiment_score']}`")
            st.caption(rec.get("explanation", ""))
            st.divider()