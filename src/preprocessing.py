import pandas as pd
import ast

movies = pd.read_csv("data/raw/ml-25m/movies.csv")
ratings = pd.read_csv("data/raw/ml-25m/ratings.csv")
links = pd.read_csv("data/raw/ml-25m/links.csv")

tmdb = pd.read_csv("data/raw/themoviesdbKaggle/movies_metadata.csv", low_memory=False)
credits = pd.read_csv("data/raw/themoviesdbKaggle/credits.csv")
keywords = pd.read_csv("data/raw/themoviesdbKaggle/keywords.csv")

tmdb = tmdb[tmdb["id"].astype(str).str.isnumeric()]
tmdb["id"] = tmdb["id"].astype(int)

credits["id"] = credits["id"].astype(int)
keywords["id"] = keywords["id"].astype(int)

def parse_top_cast(x):
    try:
        data = ast.literal_eval(x)
        return " ".join([p["name"].replace(" ", "") for p in data[:5]])
    except:
        return ""

def parse_director(x):
    try:
        data = ast.literal_eval(x)
        for p in data:
            if p.get("job") == "Director":
                return p["name"].replace(" ", "")
        return ""
    except:
        return ""

def parse_keywords(x):
    try:
        data = ast.literal_eval(x)
        return " ".join([p["name"].replace(" ", "") for p in data[:10]])
    except:
        return ""

credits["cast"] = credits["cast"].apply(parse_top_cast)
credits["director"] = credits["crew"].apply(parse_director)
keywords["keywords"] = keywords["keywords"].apply(parse_keywords)

tmdb = tmdb.merge(credits[["id", "cast", "director"]], on="id", how="left")
tmdb = tmdb.merge(keywords[["id", "keywords"]], on="id", how="left")

tmdb = tmdb[["id", "overview", "cast", "director", "keywords"]]

links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)

movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
movies = movies.merge(tmdb, left_on="tmdbId", right_on="id", how="left")

ratings = ratings.head(1000000)

active_users = ratings["userId"].value_counts()
active_users = active_users[active_users >= 20].index
ratings = ratings[ratings["userId"].isin(active_users)]

popular_movies = ratings["movieId"].value_counts()
popular_movies = popular_movies[popular_movies >= 20].index
ratings = ratings[ratings["movieId"].isin(popular_movies)]

final_df = ratings.merge(movies, on="movieId", how="left")

for col in ["genres", "overview", "cast", "director", "keywords"]:
    final_df[col] = final_df[col].fillna("")

final_df["metadata"] = (
    final_df["genres"].str.replace("|", " ", regex=False)
    + " "
    + final_df["overview"]
    + " "
    + final_df["cast"]
    + " "
    + final_df["director"]
    + " "
    + final_df["keywords"]
)

final_df.to_csv("data/processed/cleaned_movies.csv", index=False)

print("Enhanced dataset saved.")
print(final_df[["title", "genres", "cast", "director", "keywords"]].head())