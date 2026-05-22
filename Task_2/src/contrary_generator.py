VALID_SENTIMENTS = frozenset({"positive", "negative"})


def generate_contrary(support: str, sentiment: str) -> str:
    """Return the contrary for a support literal.

    positive → no_evident_not_<support>
    negative → have_evident_<support>
    """
    sent_norm = sentiment.lower().strip()
    if sent_norm not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment {sentiment!r}. Must be 'positive' or 'negative'."
        )
    if sent_norm == "positive":
        return f"no_evident_not_{support}"
    return f"have_evident_{support}"
