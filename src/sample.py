"""SAMPLE sentiment data — clearly labelled, made-up example posts.

These posts are NOT scraped. They only exist so the pipeline runs
end to end without Apify/Exa tokens. Every post they make carries
data_origin="SAMPLE", and every report/CSV says "SAMPLE SENTIMENT".

The texts are generic market chatter (no made-up facts, no made-up
prices, no made-up news). VADER scores them from the wording, the
same way it would score real posts. A fixed seed keeps the runs
the same every time.
"""
import hashlib
import random
from datetime import datetime, timedelta, timezone

TEMPLATES = [
    ("{name} looking strong this week, holding above support. "
     "Good earnings momentum, I am buying more.", "positive"),
    ("{name} is overvalued at these levels. Waiting for a dip before "
     "I touch it again.", "negative"),
    ("Holding {name} for the long term. Steady business, nothing "
     "dramatic today.", "neutral"),
    ("{name} beat expectations again. The growth story is still "
     "intact, very bullish.", "positive"),
    ("Worried about {name} — guidance was weak and the stock keeps "
     "sliding. Thinking of selling.", "negative"),
    ("{name} flat today. Volume is low, waiting for the next catalyst.",
     "neutral"),
    ("Big move in {name} after the news. Traders are excited, "
     "momentum is building fast.", "positive"),
    ("{name} dropped hard, sentiment is terrible right now. "
     "Avoid until it stabilises.", "negative"),
]

SOURCES_BY_EXCHANGE = {
    "Hong Kong": ["LIHKG", "Stocktwits", "Yahoo Finance Community"],
    "South Korea": ["Naver Finance Boards", "Daum Finance Forums",
                    "Yahoo Finance Community"],
}


def generate_sample_posts(tickers, days=10, per_ticker=8):
    """Make example posts, spread over the last `days` days."""
    now = datetime.now(timezone.utc)
    posts = []
    for t in tickers:
        # Fixed seed: Python's hash() changes every run, so I use MD5 of
        # the ticker instead — same posts every time.
        seed = int(hashlib.md5(t["ticker"].encode()).hexdigest(), 16) % 10_000
        rng = random.Random(seed)
        sources = SOURCES_BY_EXCHANGE[t["exchange"]]
        for i in range(per_ticker):
            text, _label = TEMPLATES[rng.randrange(len(TEMPLATES))]
            day_offset = rng.randrange(days)
            hour = rng.randrange(0, 24)
            ts = (now - timedelta(days=day_offset, hours=hour)).replace(
                minute=0, second=0, microsecond=0)
            posts.append({
                "ticker": t["ticker"],
                "source": "SAMPLE",
                "source_name": sources[i % len(sources)],
                "text": text.format(name=t["name"]),
                "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "likes": rng.randrange(0, 120),
                "reposts": rng.randrange(0, 40),
                "views": rng.randrange(50, 5000),
                "followers": rng.randrange(10, 5000),
                "url": "",
                "data_origin": "SAMPLE",
            })
    return posts
