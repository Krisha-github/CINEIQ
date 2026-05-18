import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

model = joblib.load("src/sentiment_model.pkl")
vectorizer = joblib.load("src/sentiment_vectorizer.pkl")
vader = SentimentIntensityAnalyzer()


def get_sentiment_score(review_text):
    if not review_text or str(review_text).strip() == "":
        return 0.5

    vader_score = vader.polarity_scores(str(review_text))["compound"]
    vader_normalized = (vader_score + 1) / 2

    try:
        vec = vectorizer.transform([str(review_text)])
        lr_score = float(model.predict_proba(vec)[0][1])
        return round((vader_normalized + lr_score) / 2, 3)
    except:
        return round(vader_normalized, 3)


def rerank_with_sentiment(recommendations):
    reranked = []

    for rec in recommendations:
        review_text = rec.get("review_text", "")

        if str(review_text).strip() == "":
            sentiment_score = rec.get("sentiment_score", 0.5)
        else:
            sentiment_score = get_sentiment_score(review_text)

        final_sentiment_score = (
            0.8 * rec["final_score"]
            + 0.2 * sentiment_score
        )

        rec["sentiment_score"] = round(sentiment_score, 3)
        rec["final_sentiment_score"] = round(final_sentiment_score, 3)

        reranked.append(rec)

    return sorted(reranked, key=lambda x: x["final_sentiment_score"], reverse=True)


if __name__ == "__main__":
    sample = [
        {
            "title": "Movie A",
            "final_score": 0.81,
            "review_text": "This movie was brilliant, emotional, and beautifully acted."
        },
        {
            "title": "Movie B",
            "final_score": 0.84,
            "review_text": "This was boring, slow, and badly written."
        }
    ]
    print(rerank_with_sentiment(sample))