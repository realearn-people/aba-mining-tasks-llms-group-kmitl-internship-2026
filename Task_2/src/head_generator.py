VALID_TOPICS = frozenset({
    "room", "staff", "location", "food",
    "price", "facility", "check-in", "check-out",
})
VALID_SENTIMENTS = frozenset({"positive", "negative"})


def generate_head(topic: str, sentiment: str) -> str:
    """Return good_<topic> or bad_<topic>.

    Topic and sentiment are normalised to lowercase before validation so
    callers can pass CSV values like 'Room' or 'Positive' unchanged.
    """
    topic_norm = topic.lower().strip()
    sent_norm = sentiment.lower().strip()

    if topic_norm not in VALID_TOPICS:
        raise ValueError(
            f"Invalid topic {topic!r}. Must be one of: {sorted(VALID_TOPICS)}"
        )
    if sent_norm not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment {sentiment!r}. Must be 'positive' or 'negative'."
        )

    prefix = "good" if sent_norm == "positive" else "bad"
    return f"{prefix}_{topic_norm}"
