"""Where the data comes from: prices (real), social posts + news (live or sample).

LIVE mode — Stocktwits public API (free, no key) and/or the X/Twitter
scraper actor on Apify (APIFY_TOKEN) plus news from exa.com (EXA_API_KEY),
all set in .env. Every record is one real post with its own UTC time,
text, likes and author follower count. X posts also carry reposts and
views; Stocktwits does not give those two, so they stay 0 there.
SAMPLE mode — no live posts at all. It scores the example posts in
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

from .config import EXA_NEWS_PER_TICKER, TICKERS
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
    """Get social posts per ticker. Gives back (posts, mode: LIVE or SAMPLE).

    LIVE now means real individual posts: Stocktwits public API (free,
    no key) plus X search through Apify (needs APIFY_TOKEN). If there is
    no Apify token I still use Stocktwits on its own — that is still
    live scraped data, not sample. Only if Stocktwits gives nothing at
    all do I fall back to the labelled example posts.
    """
    posts = _collect_stocktwits(tickers, days)
    token = os.getenv("APIFY_TOKEN")
    if token:
        posts.extend(_collect_x(tickers, days, token))
    if posts:
        return _dedupe_posts(posts), "LIVE"
    sample = _load_sample_posts(tickers, days=days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for p in sample:
        ts = datetime.fromisoformat(p["timestamp_utc"].replace("Z", "+00:00"))
        if ts >= cutoff:
            out.append({**p, "data_origin": "SAMPLE"})
    return out, "SAMPLE"


def _dedupe_posts(posts):
    """Drop exact duplicates (same source + same post URL/id)."""
    seen, out = set(), []
    for p in posts:
        key = (p.get("source_name"), p.get("url"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


STOCKTWITS_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                 "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 "
                 "Safari/537.36"}
STOCKTWITS_MAX_PAGES = 3  # 30 messages a page; more pages risks a 403


def _collect_stocktwits(tickers, days):
    """Individual Stocktwits messages via the free public API (no key).

    GET https://api.stocktwits.com/api/2/streams/symbol/<ADR>.json
    Each message has its own created_at, body, likes.total and the
    author's follower count. I check the stream's own symbol title
    first, so a wrong symbol can never slip in. Stocktwits does not
    give repost or view counts, so those stay 0 here — I say so in the
    README instead of making them up. I page back at most 3 pages and
    wait between calls, because fast repeat calls get a 403.
    """
    import time

    import requests

    from .config import STOCKTWITS_SYMBOLS

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    posts = []
    for t in tickers:
        symbol = STOCKTWITS_SYMBOLS.get(t["ticker"])
        if not symbol:
            print(f"      {t['ticker']}: Stocktwits — no symbol exists "
                  f"for {t['name']}")
            continue
        got, url_max, pages = 0, None, 0
        while pages < STOCKTWITS_MAX_PAGES:
            url = (f"https://api.stocktwits.com/api/2/streams/symbol/"
                   f"{symbol}.json")
            if url_max:
                url += f"?max={url_max}"
            try:
                resp = requests.get(url, headers=STOCKTWITS_UA, timeout=20)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Stocktwits {symbol}: request failed: {exc}")
                break
            if resp.status_code in (403, 429):
                print(f"[warn] Stocktwits {symbol}: HTTP {resp.status_code} "
                      f"(rate limit) — keeping what I have")
                break
            if resp.status_code != 200:
                print(f"[warn] Stocktwits {symbol}: HTTP {resp.status_code}")
                break
            data = resp.json()
            title = (data.get("symbol") or {}).get("title", "")
            if pages == 0:
                print(f"      {t['ticker']}: Stocktwits stream is "
                      f"'{title}' ({symbol})")
            msgs = data.get("messages", [])
            if not msgs:
                break
            for m in msgs:
                try:
                    ts = datetime.fromisoformat(
                        m["created_at"].replace("Z", "+00:00"))
                except Exception:  # noqa: BLE001
                    continue
                if ts < cutoff:
                    continue
                likes = (m.get("likes") or {}).get("total") or 0
                user = m.get("user") or {}
                body = (m.get("body") or "").strip()
                if not body:
                    continue
                posts.append({
                    "ticker": t["ticker"], "source": "Stocktwits API",
                    "source_name": "Stocktwits",
                    "text": body[:3000],
                    "timestamp_utc": m["created_at"],
                    "likes": int(likes), "reposts": 0, "views": 0,
                    "followers": int(user.get("followers") or 0),
                    "url": f"https://stocktwits.com/{symbol}/message/{m['id']}",
                    "data_origin": "LIVE",
                })
                got += 1
            pages += 1
            oldest = min(m["created_at"] for m in msgs)
            if datetime.fromisoformat(oldest.replace("Z", "+00:00")) < cutoff:
                break  # older pages are all outside the window
            url_max = (data.get("cursor") or {}).get("max")
            if not url_max or not (data.get("cursor") or {}).get("more"):
                break
            time.sleep(2.5)
        print(f"      {t['ticker']}: Stocktwits — {got} posts in the "
              f"last {days} days")
    return posts


def _collect_x(tickers, days, token):
    """Individual X posts via apisamurai/x-twitter-scraper on Apify.

    One run per ticker (the free plan allows 100 results per run, and
    I ask for 30). Each tweet row has its own createdAt, text, likes,
    retweets, views, replies and author follower count. Cost is $0.25
    per 1,000 tweets, so a full run is a few cents of the free credit.
    """
    from apify_client import ApifyClient

    from .config import APIFY_ACTORS, X_SEARCH_TERMS

    client = ApifyClient(token)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    start_date = cutoff.date().isoformat()
    # Date window goes in the search term itself (since:/until:), not in
    # the actor's endDate field. I tried endDate first: the actor pages
    # by recency and drops out-of-range tweets itself, so for a ticker
    # with a big news day today (SK Hynix, 100 tweets on 10 Nov alone)
    # it fetched 100, skipped all 100, and stopped — 0 posts. The
    # since:/until: operators filter on X's side, so the 30 I get are
    # spread over the window and land on real trading days.
    end_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    posts = []
    for t in tickers:
        term = X_SEARCH_TERMS[t["ticker"]]
        query = (f"{term} lang:en -filter:replies "
                 f"since:{start_date} until:{end_date}")
        try:
            run = client.actor(APIFY_ACTORS["x"]).call(run_input={
                "searchTerms": [query],
                "maxItems": 30,
                "startDate": start_date,
                # "Latest" sort matters: the default sort is relevance
                # ("Top"), and its results jump around between runs —
                # one full run gave me 30 SK Hynix posts all from today
                # even with until: in the query. Latest is strictly
                # newest-first inside the window, so it stays put.
                "searchSort": "Latest",
            }, logger=None)
            items = client.dataset(run.default_dataset_id).list_items().items
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] X search failed for {t['ticker']}: {exc}")
            continue
        got = 0
        for it in items:
            try:
                ts = datetime.fromisoformat(
                    str(it.get("createdAt", "")).replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                continue
            if ts < cutoff:
                continue
            # Guard: nothing dated after yesterday can line up with a
            # price bar yet, so it does not go in, even if the search
            # returned it (the default sort did this once).
            if ts.date() > (datetime.now(timezone.utc)
                            - timedelta(days=1)).date():
                continue
            text = (it.get("text") or "").strip()
            if not text:
                continue
            author = it.get("author") or {}
            posts.append({
                "ticker": t["ticker"], "source": "Apify X scraper",
                "source_name": "X (Twitter)",
                "text": text[:3000],
                "timestamp_utc": str(it.get("createdAt")),
                "likes": int(it.get("likes") or 0),
                "reposts": int(it.get("retweets") or 0),
                "views": int(it.get("views") or 0),
                "followers": int(author.get("followers") or 0),
                "url": it.get("url") or "",
                "data_origin": "LIVE",
            })
            got += 1
        print(f"      {t['ticker']}: X — {got} posts in the last "
              f"{days} days (search: '{term}')")
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
