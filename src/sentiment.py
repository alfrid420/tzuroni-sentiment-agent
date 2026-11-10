"""Sentiment scoring — VADER plus a small finance word list
(free, offline, no API key).

VADER is built for short social-media posts (slang, caps, emojis).
It gives a compound score from -1 to +1: +0.05 or more is positive,
-0.05 or less is negative, the rest is neutral. It is simple and
I can see how it works, which is the point — it is not a black box.
It can still misread sarcasm and stock-market slang.

Why not FinBERT: I tried it first in my head and checked the machine
— neither `transformers` nor `torch` is installed, and FinBERT needs
torch plus a ~400 MB model download. That is a heavy install for
387 short posts, and it would not run cleanly offline here, so I did
not force it. Instead I kept VADER and added a small finance word
list on top of it (words VADER scores as neutral but that mean
something clear in a market post, like "bullish", "downgrade" or
"beat expectations"). Each bullish word adds +0.04, each bearish
word -0.04, capped at +/-0.35 total, and the final score is clipped
to [-1, 1]. The list is short on purpose so I can still explain
every score by hand.
"""
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_ANALYSER = SentimentIntensityAnalyzer()

# Words VADER reads as neutral but that are clearly good/bad news
# for a stock. Plain words and short phrases only.
BULLISH_WORDS = [
    "bullish", "rally", "surge", "surging", "breakout", "upgrade",
    "upgraded", "outperform", "beat expectations", "beat estimates",
    "strong earnings", "record profit", "record earnings", "buy rating",
    "price target raised", "growth", "soaring", "rebound", "bull run",
    "undervalued", "buy the dip", "bulls", "moon", "all-time high",
    "dividend hike", "buyback",
]
BEARISH_WORDS = [
    "bearish", "crash", "plunge", "plunging", "downgrade", "downgraded",
    "underperform", "missed expectations", "missed estimates",
    "weak earnings", "loss", "sell-off", "selloff", "slump", "dump",
    "sell rating", "price target cut", "layoffs", "profit warning",
    "bears", "panic selling", "all-time low", "bankruptcy",
    "investigation", "fraud",
]

_WORD_RE = re.compile(r"[a-z]+")

# Each finance word shifts the score by this much; total shift capped.
FINANCE_WORD_WEIGHT = 0.04
FINANCE_SHIFT_CAP = 0.35


def _finance_shift(text_lower):
    """Count bullish minus bearish finance words/phrases in the text."""
    words = set(_WORD_RE.findall(text_lower))
    shift = 0.0
    for w in BULLISH_WORDS:
        if " " in w:
            if w in text_lower:
                shift += FINANCE_WORD_WEIGHT
        elif w in words:
            shift += FINANCE_WORD_WEIGHT
    for w in BEARISH_WORDS:
        if " " in w:
            if w in text_lower:
                shift -= FINANCE_WORD_WEIGHT
        elif w in words:
            shift -= FINANCE_WORD_WEIGHT
    return max(-FINANCE_SHIFT_CAP, min(FINANCE_SHIFT_CAP, shift))


def score_text(text):
    """Score one post or headline. Gives back (compound, label)."""
    if not text:
        return 0.0, "neutral"
    text = str(text)
    compound = _ANALYSER.polarity_scores(text)["compound"]
    compound += _finance_shift(text.lower())
    compound = max(-1.0, min(1.0, compound))
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
