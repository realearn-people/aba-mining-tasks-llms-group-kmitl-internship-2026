"""Tests for head_generator — all 16 combinations + error cases."""
import pytest
from Task_2.src.head_generator import generate_head, VALID_TOPICS

TOPICS = sorted(VALID_TOPICS)


class TestPositiveHeads:
    @pytest.mark.parametrize("topic,expected", [
        ("room",      "good_room"),
        ("staff",     "good_staff"),
        ("location",  "good_location"),
        ("food",      "good_food"),
        ("price",     "good_price"),
        ("facility",  "good_facility"),
        ("check-in",  "good_check-in"),
        ("check-out", "good_check-out"),
    ])
    def test_positive(self, topic, expected):
        assert generate_head(topic, "positive") == expected

    def test_positive_case_insensitive_topic(self):
        assert generate_head("Room", "positive") == "good_room"

    def test_positive_case_insensitive_sentiment(self):
        assert generate_head("staff", "Positive") == "good_staff"


class TestNegativeHeads:
    @pytest.mark.parametrize("topic,expected", [
        ("room",      "bad_room"),
        ("staff",     "bad_staff"),
        ("location",  "bad_location"),
        ("food",      "bad_food"),
        ("price",     "bad_price"),
        ("facility",  "bad_facility"),
        ("check-in",  "bad_check-in"),
        ("check-out", "bad_check-out"),
    ])
    def test_negative(self, topic, expected):
        assert generate_head(topic, "negative") == expected

    def test_negative_case_insensitive(self):
        assert generate_head("Staff", "Negative") == "bad_staff"


class TestErrorCases:
    def test_invalid_topic_raises(self):
        with pytest.raises(ValueError, match="Invalid topic"):
            generate_head("weather", "positive")

    def test_invalid_sentiment_raises(self):
        with pytest.raises(ValueError, match="Invalid sentiment"):
            generate_head("room", "neutral")

    def test_empty_topic_raises(self):
        with pytest.raises(ValueError, match="Invalid topic"):
            generate_head("", "positive")

    def test_empty_sentiment_raises(self):
        with pytest.raises(ValueError, match="Invalid sentiment"):
            generate_head("room", "")

    def test_off_topic_invalid(self):
        """'Off' rows are filtered before reaching Task 2."""
        with pytest.raises(ValueError):
            generate_head("off", "positive")
