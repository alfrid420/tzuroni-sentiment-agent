"""Where the data comes from: prices (real), social posts + news (live or sample).

LIVE mode — Apify token (APIFY_TOKEN) and/or Exa key (EXA_API_KEY) are
set in .env. Social posts come from Apify actors and news comes from
exa.com. Every record keeps its source, URL and UTC time.
SAMPLE mode — no tokens set. It scores the example posts in
src/sample.py, so the whole thing still runs end to end. Sample posts
are NEVER passed off as scraped posts: `source` is "SAMPLE" and every
output carries a "SAMPLE SENTIMENT" note.

Prices are always REAL: daily prices from Yahoo Finance (yfinance),
free and no key. If the network is down the run fails loudly instead
of making up prices.
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf

from .config import APIFY_ACTORS, EXA_NEWS_PER_TICKER, TICKERS
from .sentiment import score_text

ROOT = Path(__file__).resolve().parent.parent


def fetch_prices(tickers, days=90):
    """Real daily prices from Yahoo Finance. Gives back {symbol: table}."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days + 10)
    out = {}
    for t in tickers:
        df = yf.download(t["yahoo"], start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(f"No price data for {t['yahoo']} — check network / ticker")
        # yfinance sometimes gives multi-level columns, so flatten them.
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Date"] = df["Date"].dt.tz_localize(None)
        out[t["yahoo"]] = df
    return out


def _load_sample_posts(tickers, days=10):
    from .sample import generate_sample_posts

    return generate_sample_posts(tickers, days=days)


def collect_social(tickers, days=10):
    """Get social posts per ticker. Gives back (posts, mode: LIVE or SAMPLE)."""
    token = os.getenv("APIFY_TOKEN")
    if token:
        return _collect_social_live(tickers, days, token), "LIVE"
    posts = []
    sample = _load_sample_posts(tickers, days=days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    for p in sample:
        ts = datetime.fromisoformat(p["timestamp_utc"].replace("Z", "+00:00"))
        if ts >= cutoff:
            posts.append({**p, "data_origin": "SAMPLE"})
    return posts, "SAMPLE"


def _collect_social_live(tickers, days, token):
    """Live Apify collection (free tier) with website-content-crawler.

    One actor run per ticker, crawling its discussion pages (LIHKG /
    Stocktwits / Yahoo for HK; Naver / Daum / Yahoo for KR). Each crawled
    page becomes one record — the page text is what people are saying
    about the stock on that page. I print what worked and what did not,
    per ticker and per source, and I never fall back to sample data here.

    One honest limit: a page crawl gives me text, not per-post likes,
    reposts, views or follower counts, so those fields are 0 in LIVE
    mode. The README says this too.
    """
    from apify_client import ApifyClient

    from .config import discussion_urls

    client = ApifyClient(token)
    posts = []
    now_utc = datetime.now(timezone.utc).isoformat()
    for t in tickers:
        urls = discussion_urls(t)
        try:
            run = client.actor(APIFY_ACTORS["web"]).call(run_input={
                "startUrls": [{"url": u} for _, u in urls],
                "maxCrawlPages": len(urls),
                "maxCrawlDepth": 0,
                "crawlerType": "playwright:adaptive",
            }, logger=None)
            items = client.dataset(run.default_dataset_id).list_items().items
        except Exception as exc:  # noqa: BLE001 — report, don't crash the run
            print(f"[warn] Apify crawl failed for {t['ticker']}: {exc}")
            continue
        got = {}
        for it in items:
            url = it.get("url", "")
            text = (it.get("text") or it.get("markdown") or "").strip()
            source_name = next(
                (name for name, u in urls if u in url), url)
            if len(text) < 100:
                print(f"[warn] {t['ticker']} {source_name}: page came back "
                      f"empty or too short ({len(text)} chars) — skipped")
                continue
            got[source_name] = got.get(source_name, 0) + 1
            ts = it.get("date") or it.get("timestamp") or now_utc
            posts.append({
                "ticker": t["ticker"], "source": "Apify",
                "source_name": source_name,
                "text": text[:3000],
                "timestamp_utc": ts,
                "likes": 0, "reposts": 0, "views": 0, "followers": 0,
                "url": url,
                "data_origin": "LIVE",
            })
        if got:
            print(f"      {t['ticker']}: LIVE pages — " +
                  ", ".join(f"{s} x{n}" for s, n in got.items()))
        else:
            print(f"[warn] {t['ticker']}: no usable LIVE pages at all")
    return posts


def collect_news(tickers):
    """Recent news per ticker from Exa (free tier). Gives back (news, mode)."""
    key = os.getenv("EXA_API_KEY")
    if not key:
        return [], "SAMPLE"
    from exa_py import Exa

    exa = Exa(api_key=key)
    news = []
    for t in tickers:
        try:
            res = exa.search_and_contents(
                f"{t['name']} stock news", num_results=EXA_NEWS_PER_TICKER,
                type="auto", category="news", text=True)
            for r in res.results:
                news.append({
                    "ticker": t["ticker"], "source": "Exa",
                    "source_name": "exa.com",
                    "title": r.title, "text": (r.text or "")[:500],
                    "url": r.url, "published": str(r.published_date or ""),
                    "data_origin": "LIVE",
                })
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Exa failed for {t['ticker']}: {exc}")
    return news, "LIVE"


def score_posts(posts):
    """Score every post with VADER and add the score + label."""
    scored = []
    for p in posts:
        compound, label = score_text(p.get("text", ""))
        scored.append({**p, "sentiment": compound, "sentiment_label": label})
    return scored


def score_news(news):
    scored = []
    for n in news:
        compound, label = score_text(f"{n.get('title', '')}. {n.get('text', '')}")
        scored.append({**n, "sentiment": compound, "sentiment_label": label})
    return scored
