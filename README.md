# HengSeng Sentiment Price Prediction Agent

My intern assessment for **TZURONI / CrowdWisdomTrading** (Gilad, CEO), through Internshala.
**Deadline: 15 November 2026.** I will submit it by email to `gilad@crowdwisdomtrading.com` with (1) a GitHub repo link and (2) the Apify/Exa tokens + actor IDs, so they can run it themselves.

## What I built

I built a Python pipeline that tries to guess each stock's price **tomorrow (n+1)** from what people are saying about it online.

| Exchange | Tickers (5 each) |
|---|---|
| Hong Kong | 0700 Tencent · 9988 Alibaba · 3690 Meituan · 9618 JD.com · 1810 Xiaomi |
| South Korea | 005930 Samsung Electronics · 000660 SK Hynix · 035420 Naver · 005380 Hyundai Motor · 051910 LG Chem |

- **Sentiment sources (2 or more per exchange, like the task asked):** the Stocktwits public API (free, no key) for the US ADR of each stock, plus X (Twitter) search through the Apify actor `apisamurai/x-twitter-scraper` — plus recent news from **exa.com**. Both give me real individual posts with their own timestamps, text, likes and author follower counts. I dropped the discussion-page crawl (LIHKG / Naver / Daum / Yahoo Community): a page crawl returns page text, not individual dated posts, so likes, reposts, views and followers were all 0 and nothing lined up with price days. I dropped Telegram too: there is no official Apify actor called `apify/telegram-scraper`, I checked, so I did not want to use a made-up ID. Reddit's public pages answer 403 without a login now, so that did not work either.
- **Prices:** real daily prices from Yahoo Finance (free, no key needed).
- **Sentiment:** VADER plus a small finance word list (bullish, rally, downgrade, beat expectations, sell-off — each word shifts the score by 0.04, capped at 0.35). I did not use FinBERT: it needs torch plus a ~400 MB model download, neither was installed, and that is a heavy install for a few hundred short posts. VADER plus the word list runs offline for free and I can explain every score by hand.
- **Model:** I compare Ridge, RandomForest and GradientBoosting (scikit-learn only) on the same time split, and take the best per ticker by test directional accuracy. Each model's settings were picked on a validation split inside the training period only, never on the test set. Features per day: mean sentiment, 3-day sentiment, mention count, log-engagement, news sentiment, 1-day and 5-day returns, log volume — plus RSI (14d), MACD and its histogram, price vs the 5/10/20-day averages, 5-day and 20-day average slopes, Bollinger position, volume change vs its 20-day average, 5-day volatility, day of the week. All of them use prices up to today only.
- **Testing:** I split by time — train on the first 70% of days, test on the last 30%. No shuffling, because shuffling would let the model peek at the future. I only report the test numbers. I also report a baseline that just says tomorrow moves the same way as yesterday, so I can see whether a model is really learning anything.
- **Per stock:** current price, sentiment, direction (Bullish / Bearish / Neutral), support and resistance (20-day low / high), and the n+1 prediction.
- **Report:** `report/report.html` — tables sorted by mentions / sentiment / direction for each exchange, plus charts.

## Be honest about the data — read this first

- **Prices are real** (Yahoo Finance, fetched when I run it).
- **Social posts and news are SAMPLE unless I set tokens.** Without tokens the pipeline runs in **SAMPLE mode**: it scores a small set of example posts (`src/sample.py`) so everything still runs end to end. I never pass sample posts off as scraped posts — every CSV and the report carry a `SAMPLE SENTIMENT` note in that mode.
- With free tokens in `.env` the collectors switch to **LIVE mode** and pull real posts and news. Tokens are never written in the code and never committed to git. The Stocktwits API needs no token at all, so even without an Apify token the run is still live for the 7 tickers that have a Stocktwits stream.

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
| `data/model_comparison.csv` | Ridge vs RandomForest vs GradientBoosting, test numbers per ticker (all three, not just the winner) |
| `report/report.html` | The visual report |
| `report/mentions_sentiment.png` | Mentions per ticker, coloured by sentiment |
| `report/price_prediction.png` | 30-day price history + n+1 prediction per ticker |

## Free tokens (for LIVE mode — I need these before I submit)

1. **Apify** — sign up free at https://apify.com, copy your token from Settings → Integrations. Free tier has trial credits. Actor I used (public ID, not a secret): `apisamurai/x-twitter-scraper` — it charges $0.25 per 1,000 tweets, so a full run costs a few cents.
2. **Exa** — sign up free at https://exa.ai, copy your API key from the dashboard. The free tier is enough for 5 news items × 10 tickers.

Put them in `.env` (never in code, never in git):

```
APIFY_TOKEN=your_free_apify_token
EXA_API_KEY=your_free_exa_key
```

## What is wrong with it (honest notes)

- Direction accuracy is near 50%. That means the model is barely better than guessing up or down for a 1-day move from sentiment alone. I tried to fix that on 10 Nov (see the run note below) and it did not really work — I am reporting it as it is.
- VADER is just a word list, not a finance model. It can misread sarcasm and stock-market slang. The finance word list helps a bit on words like "downgrade" and "beat expectations", but it does not turn it into a finance model.
- Prices now cover 1 year, but social posts only cover the last 10 days, so sentiment is 0 for most of the year and the technical features do most of the work on the older days. That is a real limit of this data, not something I can fix by fetching more prices.
- The n+1 point forecast is capped at +/-5%. The test numbers are NOT capped. On 10 Nov the uncapped Ridge guess for Samsung was -10.7% during a sharp sell-off — a linear model extrapolating, not a forecast I would stand behind — so the reported Samsung n+1 is -5.00%, the capped value.
- In LIVE mode every post is one real post with its own UTC date, and the posts feed the model on their own dates. A few limits I should say out loud: Stocktwits has no stream at all for SK Hynix, Naver or LG Chem, and Meituan's stream has not had a post since 8 Sep 2026, so those tickers only get X posts. Xiaomi and Hyundai have a Stocktwits stream but it is quiet — 2 and 1 posts each in the last 10 days. Stocktwits does not give repost or view counts, so those are 0 for its posts. X search is in English only, so it misses what Korean and Chinese traders write in their own languages. And X counts move around between runs: the search returns the latest posts first, so a busy news day can fill the 30-post cap with one day's posts. I cap X at 30 posts per ticker and search with a "Latest" sort and since:/until: dates, so the posts land on trading days.
- Bare numbers are a trap on Stocktwits: "005930" is the IWM ETF there, not Samsung, and "000660" is Treasury futures, not SK Hynix. I only use a symbol when the stream's own title names the company, and the collector checks that title on every run.
- "Position: No position held" — this is an assessment, not trading advice. I made no real trades.
- Not investment advice.

## LIVE run — 10 Nov 2026 (what actually came back)

I fixed the social-sentiment gap on 10 Nov: the pipeline used to crawl whole discussion pages, so every ticker showed 0 mentions and 0.00 sentiment. Now it pulls individual posts.

- Stocktwits public API (free, no key): 210 posts in the last 10 days. Alibaba 90 and JD.com 90 (capped at 3 pages each), Samsung 15, Tencent 12, Xiaomi 2, Hyundai 1, Meituan 0 (its stream is dead since 8 Sep 2026). No stream exists for SK Hynix, Naver or LG Chem. Each message carries its own timestamp, text, likes and author followers.
- X via Apify actor `apisamurai/x-twitter-scraper` ($0.25 per 1,000 tweets; this run cost a few cents of the free credit): 177 posts, 30 each for Tencent, Alibaba, Meituan, Xiaomi and Samsung, 14 for Naver, 10 for SK Hynix, 8 for Hyundai, 6 for JD.com, 2 for LG Chem. Each tweet carries its own timestamp, text, likes, reposts, views and author followers.
- Total: 387 live posts, all dated 31 Oct – 10 Nov 2026. 296 have likes above 0, 385 have followers above 0, 177 have views above 0. Every ticker now has non-zero mentions and non-zero mean sentiment: Tencent 32 mentions / +0.21, Alibaba 85 / +0.10, Meituan 30 / +0.03, JD.com 86 / +0.04, Xiaomi 27 / +0.17, Samsung 28 / +0.12, SK Hynix 10 / +0.02, Naver 12 / +0.34, Hyundai 7 / +0.25, LG Chem 2 / +0.05.
- Exa news: 50 live items, 5 per ticker, all dated (Nov 5–6, 2026).
- Prices: real, Yahoo Finance, up to 9 Nov 2026.
- Mean test directional accuracy: about 50% — barely better than guessing, reported as it is.
- What still does not work: Naver / Daum / LIHKG boards have no free per-post source I could verify, so Korean and Hong Kong retail chatter in Korean and Chinese is missing — X search is English only. Reddit is 403 without a login. The older run's 19 crawled pages are gone from the outputs; nothing in the new run comes from a page crawl.

## LIVE run — 10 Nov 2026, second run (I tried to make it more accurate)

I changed four things and re-ran the whole pipeline end to end (exit 0, LIVE mode, 374 posts, 50 news items, prices 31 Oct 2025 – 9 Nov 2026):

1. **Longer price history:** 1 year instead of 90 days. Each ticker now has 157–158 training days and 68–69 test days, instead of about 40 training days and 19 test days. That alone makes the numbers more trustworthy — 19 test days is too few to tell anything.
2. **More features:** RSI (14d), MACD and its histogram, price vs the 5/10/20-day averages, average slopes, Bollinger position, volume change, 5-day volatility, day of the week. All from prices up to today only.
3. **Better model:** Ridge vs RandomForest vs GradientBoosting on the same time split, settings picked on a validation split inside the training period only.
4. **Better sentiment:** VADER plus a small finance word list. I did not use FinBERT — torch and transformers were not installed, and it needs a ~400 MB model download. That is a heavy install for 374 short posts, so I kept it offline and free and said so.

Test directional accuracy per ticker (last 30% of days, 68–69 days each):

| Ticker | Ridge | RandomForest | GradientBoosting | Best | Baseline (yesterday repeats) |
|---|---|---|---|---|---|
| 0700 Tencent | 61% | 55% | 54% | Ridge 61% | 55% |
| 9988 Alibaba | 57% | 54% | 51% | Ridge 57% | 43% |
| 3690 Meituan | 41% | 55% | 48% | RandomForest 55% | 42% |
| 9618 JD.com | 54% | 57% | 55% | RandomForest 57% | 35% |
| 1810 Xiaomi | 52% | 48% | 46% | Ridge 52% | 51% |
| 005930 Samsung | 44% | 40% | 38% | Ridge 44% | 40% |
| 000660 SK Hynix | 50% | 47% | 47% | Ridge 50% | 51% |
| 035420 Naver | 54% | 57% | 59% | GradientBoosting 59% | 53% |
| 005380 Hyundai | 56% | 49% | 53% | Ridge 56% | 44% |
| 051910 LG Chem | 46% | 53% | 51% | RandomForest 53% | 44% |

Means across all 10 tickers: Ridge 51.4%, RandomForest 51.4%, GradientBoosting 50.2%, baseline 45.8%. If I take the best model per ticker the mean is 54.3% — but that number is flattered, because I picked the winner after seeing the test set, so I do not count it as a real gain.

**Did it help? Honestly, no, not in a way I can stand behind.** The old run was Ridge at 50.3% on about 19 test days per ticker. The new run is Ridge at 51.4% on about 69 test days per ticker. That is +1 point on a much bigger test set, and with 69 test days the noise on one ticker's accuracy is about +/-6 points, so +1 is noise. RandomForest and GradientBoosting are no better than Ridge — the trees just fit the noise in a different way. The models do beat the "yesterday repeats" baseline (45.8%), but only by a bit, and two tickers (Samsung 44%, SK Hynix 50%) are at or below guessing. Samsung is the worst: it fell from about 285,500 to 188,700 in a few weeks, its daily moves are huge (test MAE 0.040, RMSE 0.056), and no model here can call that.

What I take from this: a 1-day move is mostly noise, and more history and more technical features do not change that — they just make the near-50% answer more solid. The sentiment data is the real weak spot: it only covers 10 days, so for most of the year the model has no sentiment signal at all. Fixing that would mean collecting posts every day for months, not a better model. I did not overfit to chase a higher number, I did not touch the test set while tuning, and I am keeping the 50–55% numbers as they are.

Two things that changed in this run that I should also say out loud: SK Hynix got 0 posts this time (X returned 0 for "SK Hynix", and it has no Stocktwits stream), so its sentiment is 0.00 — X counts move around between runs, that is not a bug I introduced. And the Samsung n+1 prediction is capped at -5.00% (uncapped it was -10.7%, see the note above).

## Project structure

```
main.py                  runs the whole pipeline
src/config.py            tickers, sources, Apify actors, windows
src/collectors.py        prices (yfinance) + social (Stocktwits API + X via Apify) + news (Exa)
src/sample.py            example posts, clearly labelled, not scraped
src/sentiment.py         VADER scoring + finance word list + direction labels
src/model.py             features (sentiment + technicals), support/resistance,
                         Ridge vs RandomForest vs GradientBoosting, testing
src/report.py            charts + HTML report
```

Built by Monu Mishra (BCA 2026) — Python, scikit-learn, NLP.
