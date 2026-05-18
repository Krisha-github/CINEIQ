import pandas as pd

movies = pd.read_csv("data/raw/ml-25m/movies.csv")
ratings = pd.read_csv("data/raw/ml-25m/ratings.csv")

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

print("\nMovies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())