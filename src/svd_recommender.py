import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

df = pd.read_csv("data/processed/cleaned_movies.csv")

df = df.head(100000)

ratings_data = df[["userId", "movieId", "rating"]]

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings_data,
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)

print("\nRMSE:", rmse)

movies = df[["movieId", "title"]].drop_duplicates()


def recommend_movies_svd(user_id, top_n=10):

    watched_movies = df[df["userId"] == user_id]["movieId"].unique()

    unwatched_movies = movies[
        ~movies["movieId"].isin(watched_movies)
    ]

    recommendations = []

    for _, row in unwatched_movies.iterrows():

        movie_id = row["movieId"]
        title = row["title"]

        predicted_rating = model.predict(
            user_id,
            movie_id
        ).est

        recommendations.append(
            (title, predicted_rating)
        )

    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:top_n]


print("\nTop recommendations for User 1:\n")

recommendations = recommend_movies_svd(1)

for movie, score in recommendations:
    print(f"{movie} --> Predicted Rating: {score:.2f}")