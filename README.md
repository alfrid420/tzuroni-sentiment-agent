# HengSeng Sentiment Price Prediction Agent

My intern assessment for **TZURONI / CrowdWisdomTrading** (Gilad, CEO), through Internshala.
**Deadline: 15 November 2026.** I will submit it by email to `gilad@crowdwisdomtrading.com` with (1) a GitHub repo link and (2) the Apify/Exa tokens + actor IDs, so they can run it themselves.

## What I built

I built a Python pipeline that tries to guess each stock's price **tomorrow (n+1)** from what people are saying about it online.

| Exchange | Tickers (5 each) |
|---|---|
| Hong Kong | 0700 Tencent · 9988 Alibaba · 3690 Meituan · 9618 JD.com · 1810 Xiaomi |
| South Korea | 005930 Samsung Electronics · 000660 SK Hynix · 035420 Naver · 005380 Hyundai Motor · 051910 LG Chem |

- **Sentiment sources (2 or more per exchange, like the task asked):** Apify's website-content-crawler — LIHKG / Stocktwits / Yahoo Community for Hong Kong; Naver / Daum / Yahoo Community for Korea — plus recent news from **exa.com**. I dropped Telegram: there is no official Apify actor called `apify/telegram-scraper`, I checked, so I did not want to use a made-up ID.
- **Prices:** real daily prices from Yahoo Finance (free, no key needed).
- **Sentiment:** VADER — a simple word list, works offline, no API.
- **Model:** Ridge regression (scikit-learn). Features per day: mean sentiment, 3-day sentiment, mention count, log-engagement, news sentiment, 1-day and 5-day returns, log volume.
- **Testing:** I split by time — train on the first 70% of days, test on the last 30%. No shuffling, because shuffling would let the model peek at the future. I only report the test numbers.
- **Per stock:** current price, sentiment, direction (Bullish / Bearish / Neutral), support and resistance (20-day low / high), and the n+1 prediction.
- **Report:** `report/report.html` — tables sorted by mentions / sentiment / direction for each exchange, plus charts.

## Be honest about the data — read this first

- **Prices are real** (Yahoo Finance, fetched when I run it).
- **Social posts and news are SAMPLE unless I set tokens.** Without tokens the pipeline runs in **SAMPLE mode**: it scores a small set of example posts (`src/sample.py`) so everything still runs end to end. I never pass sample posts off as scraped posts — every CSV and the report carry a `SAMPLE SENTIMENT` note in that mode.
- With free tokens in `.env` the collectors switch to **LIVE mode** and scrape real posts and news. Tokens are never written in the code and never committed to git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# SAMPLE mode (no tokens needed — runs end to end, sentiment is labelled sample)
python main.py

# LIVE mode (free tiers only — I paid nothing)
cp .env.example .env   # then fill in your free tokens
python main.py
```

Open `report/report.html` in a browser to see the report.

## Outputs

| File | What is inside |
|---|---|
| `data/raw_prices.csv` | Real daily prices, all 10 tickers |
| `data/posts_scored.csv` | Scored social posts (with `data_origin`: LIVE / SAMPLE) |
| `data/news_scored.csv` | Scored news (LIVE mode; empty in SAMPLE mode) |
| `data/daily_features.csv` | Daily features + n+1 target, all tickers |
| `data/predictions.csv` | n+1 predictions, support/resistance, test numbers |
| `report/report.html` | The visual report |
| `report/mentions_sentiment.png` | Mentions per ticker, coloured by sentiment |
| `report/price_prediction.png` | 30-day price history + n+1 prediction per ticker |

## Free tokens (for LIVE mode — I need these before I submit)

1. **Apify** — sign up free at https://apify.com, copy your token from Settings → Integrations. Free tier has trial credits. Actor I used (public ID, not a secret): `apify/website-content-crawler`.
2. **Exa** — sign up free at https://exa.ai, copy your API key from the dashboard. The free tier is enough for 5 news items × 10 tickers.

Put them in `.env` (never in code, never in git):

```
APIFY_TOKEN=your_free_apify_token
EXA_API_KEY=your_free_exa_key
```

## What is wrong with it (honest notes)

- Direction accuracy is near 50%. That means the model is barely better than guessing up or down for a 1-day move from sentiment alone. I am reporting it as it is.
- VADER is just a word list, not a finance model. It can misread sarcasm and stock-market slang.
- In LIVE mode each crawled discussion page counts as one record, and the page crawl does not give me per-post likes, reposts, views or follower counts, so those are 0. The sentiment score comes from the page text itself.
- Live social pages are dated at crawl time, so they do not line up with past price days — that is why the report shows 0 mentions. The live news (50 dated items from Exa) does line up and does feed the model.
- "Position: No position held" — this is an assessment, not trading advice. I made no real trades.
- Not investment advice.

## LIVE run — 10 Nov 2026 (what actually came back)

- Apify website-content-crawler: 19 live pages. Hong Kong — Stocktwits and Yahoo Community worked for all 5 tickers; LIHKG worked only for Xiaomi (its search pages came back nearly empty for the other 4). South Korea — Naver boards worked for Samsung, Naver and LG Chem; Daum discussion pages came back nearly empty for all 5; Yahoo Community worked for all 5. SK Hynix and Hyundai only got Yahoo Community.
- Exa news: 50 live items, 5 per ticker, all dated (Nov 5–6, 2026).
- Prices: real, Yahoo Finance, up to 9 Nov 2026.
- Mean test directional accuracy: about 50% — barely better than guessing, reported as it is.

## Project structure

```
main.py                  runs the whole pipeline
src/config.py            tickers, sources, Apify actors, windows
src/collectors.py        prices (yfinance) + social (Apify) + news (Exa)
src/sample.py            example posts, clearly labelled, not scraped
src/sentiment.py         VADER scoring + direction labels
src/model.py             features, support/resistance, Ridge, testing
src/report.py            charts + HTML report
```

Built by Monu Mishra (BCA 2026) — Python, scikit-learn, NLP.
