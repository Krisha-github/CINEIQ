# CINEIQ — Explainable Hybrid Movie Recommendation Engine

CINEIQ is an end-to-end movie recommendation system built with Python that combines multiple ML strategies to deliver personalized, interpretable movie suggestions. Every recommendation comes with a human-readable explanation of why it was suggested.

---

## Project Structure

```
CINEIQ/
├── app/
│   ├── api.py                        # FastAPI serving endpoints
│   └── dashboard.py                  # Streamlit user dashboard
├── src/
│   ├── checkdata.py                  # Data inspection script
│   ├── collaborative_recommender.py  # User-user collaborative filtering
│   ├── content_recommender.py        # TF-IDF + cosine similarity
│   ├── explainability.py             # Rule-based explanation generator
│   ├── hybrid_recommender.py         # Weighted ensemble of all three methods
│   ├── mlflow_tracking.py            # Experiment tracking with MLflow
│   ├── preprocessing.py              # Data cleaning and feature engineering
│   ├── sentiment.py                  # VADER + Logistic Regression reranker
│   ├── svd_recommender.py            # SVD matrix factorization
│   └── train_sentiment_model.py      # Trains sentiment model on IMDB 50K                      
├── requirements.txt
└── README.md
```

---

## Deliverables

### 1. Hybrid Recommendation Engine
Combines three strategies into a weighted ensemble:

| Component | Method | Weight |
|---|---|---|
| Content-Based | TF-IDF + Cosine Similarity | 25% |
| Collaborative | User-User Cosine Similarity | 15% |
| Matrix Factorization | SVD via Surprise | 40% |
| Sentiment | VADER + Logistic Regression | 20% |

Metadata used: genres, overview, cast, director, keywords from TMDB.

---

### 2. Sentiment-Aware Re-Ranker
Trained a binary sentiment classifier on the IMDB 50K reviews dataset using TF-IDF vectorization and Logistic Regression. VADER is used alongside it for real-time sentiment scoring. Recommendations are reranked using an 80/20 blend of hybrid score and sentiment score.

---

### 3. User Taste Dashboard
Built with Streamlit and Plotly. Visualizes:
- Genre radar chart (average rating per genre)
- Decade preferences (rating trend across decades)
- Director affinity (top 10 directors by average rating)
- Actor affinity (top 10 actors by average rating)
- Hybrid recommendations with explanations

---

### 4. Explainability Layer
Every recommendation includes a rule-based human-readable explanation. Example output:

```
Recommended because it matches your preferred genres, cast, director, or keywords,
the SVD model predicts you would rate it highly, and it is directed by ChristopherNolan.
```

---

## Datasets

| Dataset | Source | Usage |
|---|---|---|
| MovieLens 25M | grouplens.org/datasets/movielens/25m | Ratings and movie metadata |
| TMDB Metadata | Kaggle | Cast, genres, keywords for 45K movies |
| IMDB 50K Reviews | Kaggle | Sentiment model training |

> Datasets are not included in this repository. Download and place them in `data/raw/` as described in Setup Instructions.

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | Python, scikit-learn, Surprise (SVD), Pandas, NumPy |
| NLP | VADER, TF-IDF, Logistic Regression |
| Serving | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Tracking | MLflow |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Krisha-github/CINEIQ.git
cd CINEIQ
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- MovieLens 25M → place contents in `data/raw/ml-25m/`
- TMDB Metadata → place contents in `data/raw/themoviesdbKaggle/`
- IMDB 50K Reviews → place file in `data/raw/imdb/`

### 5. Run preprocessing
```bash
python src/preprocessing.py
```

### 6. Train sentiment model
```bash
python src/train_sentiment_model.py
```

### 7. Run the API
```bash
uvicorn app.api:app --reload
```

### 8. Run the dashboard
```bash
streamlit run app/dashboard.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/recommend/{user_id}` | Get hybrid recommendations for a user |
| GET | `/similar?movie_title=` | Get content-similar movies |

Example:
```
GET /recommend/1?top_n=10
GET /similar?movie_title=Toy Story (1995)
```

---

## Experiment Tracking

MLflow is used to log SVD hyperparameters and RMSE across runs.

```bash
mlflow ui
```

Open `http://localhost:5000` to view experiment history.

---

## Example Recommendation Output

```json
{
  "title": "Inception",
  "final_sentiment_score": 0.874,
  "explanation": "Recommended because it matches your preferred genres, the SVD model predicts you would rate it highly, audience review sentiment is positive, and it is directed by ChristopherNolan."
}
```

---

## Author

Built by Krisha Kapadia demonstrating end-to-end ML system design, NLP, API development, and data visualization.
