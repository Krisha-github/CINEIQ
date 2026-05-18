import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/cleaned_movies.csv")

movies = df[["movieId", "title", "genres", "metadata"]].drop_duplicates().reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
tfidf_matrix = tfidf.fit_transform(movies["metadata"].fillna(""))

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend_similar_movies(title, top_n=10):
    if title not in indices:
        return f"Movie '{title}' not found."

    idx = indices[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    movie_indices = [i[0] for i in scores]

    return movies.iloc[movie_indices][["title", "genres"]]

print(recommend_similar_movies("Toy Story (1995)", 10))