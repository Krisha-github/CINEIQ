import mlflow
import mlflow.sklearn

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

with mlflow.start_run():

    # Hyperparameters
    n_factors = 100
    n_epochs = 20

    # Log parameters
    mlflow.log_param("n_factors", n_factors)
    mlflow.log_param("n_epochs", n_epochs)

    # Train model
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs
    )

    model.fit(trainset)

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)

    mlflow.log_metric("RMSE", rmse)

    print("\nModel tracked successfully!")