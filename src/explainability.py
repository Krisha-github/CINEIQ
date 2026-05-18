def generate_explanation(rec):
    reasons = []

    if rec.get("content_score", 0) >= 0.3:
        reasons.append("it matches your preferred genres, cast, director, or keywords")

    if rec.get("collaborative_score", 0) >= 0.6:
        reasons.append("users with similar taste rated it highly")

    if rec.get("svd_score", 0) >= 0.8:
        reasons.append("the SVD model predicts you would rate it highly")

    if rec.get("sentiment_score", 0.5) >= 0.7:
        reasons.append("audience review sentiment is positive")

    director = rec.get("director", "")
    if director:
        reasons.append(f"it is directed by {director}")

    cast = rec.get("cast", "")
    if cast:
        reasons.append(f"it features actors such as {cast}")

    if not reasons:
        reasons.append("it is a balanced recommendation based on your rating history")

    return "Recommended because " + ", ".join(reasons) + "."