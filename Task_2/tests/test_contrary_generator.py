"""Tests for contrary_generator."""
import pytest
from Task_2.src.contrary_generator import generate_contrary


class TestPositiveContraries:
    def test_basic_positive(self):
        assert generate_contrary("clean_room", "positive") == "no_evident_not_clean_room"

    def test_multiword_support_positive(self):
        assert (
            generate_contrary("helpful_staff", "positive")
            == "no_evident_not_helpful_staff"
        )

    def test_positive_case_insensitive(self):
        assert generate_contrary("tasty_food", "Positive") == "no_evident_not_tasty_food"

    @pytest.mark.parametrize("support", [
        "new_hotel", "comfortable_hotel", "close_to_airport",
        "easy_to_communicate_with", "tasty_food",
    ])
    def test_positive_prefix(self, support):
        result = generate_contrary(support, "positive")
        assert result.startswith("no_evident_not_")
        assert result.endswith(support)


class TestNegativeContraries:
    def test_basic_negative(self):
        assert generate_contrary("not_big_room", "negative") == "have_evident_not_big_room"

    def test_negative_case_insensitive(self):
        assert generate_contrary("slow_elevator", "Negative") == "have_evident_slow_elevator"

    @pytest.mark.parametrize("support", [
        "not_big_room", "expensive_price", "noisy_room",
    ])
    def test_negative_prefix(self, support):
        result = generate_contrary(support, "negative")
        assert result.startswith("have_evident_")
        assert result.endswith(support)


class TestErrorCases:
    def test_invalid_sentiment_raises(self):
        with pytest.raises(ValueError, match="Invalid sentiment"):
            generate_contrary("clean_room", "neutral")

    def test_empty_sentiment_raises(self):
        with pytest.raises(ValueError):
            generate_contrary("clean_room", "")
