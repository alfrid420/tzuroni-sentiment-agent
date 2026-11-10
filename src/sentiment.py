"""Sentiment scoring — VADER, a word list (free, offline, no API key).

VADER is built for short social-media posts (slang, caps, emojis).
It gives a compound score from -1 to +1: +0.05 or more is positive,
-0.05 or less is negative, the rest is neutral. It is simple and
I can see how it works, which is the point — it is not a black box.
It can still misread sarcasm and stock-market slang.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYSER = SentimentIntensityAnalyzer()


def score_text(text):
    """Score one post or headline. Gives back (compound, label)."""
    if not text:
        return 0.0, "neutral"
    compound = _ANALYSER.polarity_scores(str(text))["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


def direction_from_sentiment(mean_compound):
    """Bullish / Bearish / Neutral, from the mean score."""
    if mean_compound >= 0.05:
        return "Bullish"
    if mean_compound <= -0.05:
        return "Bearish"
    return "Neutral"
