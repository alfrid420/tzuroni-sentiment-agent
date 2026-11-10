"""CWT / TZURONI — HengSeng Sentiment Price Prediction Agent.

My intern assessment (Gilad, CrowdWisdomTrading CEO, through Internshala,
Nov 2026). Deadline: 15 Nov 2026.

What it does: guesses each stock's price tomorrow (n+1) from social
sentiment + news sentiment.

* Hong Kong and South Korea — 5 stocks each.
* Sentiment sources (2 or more per exchange): the Stocktwits public
  API (free, no key) for each stock's US ADR, plus X (Twitter) search
  through the Apify actor apisamurai/x-twitter-scraper, plus news
  from exa.com. Each source gives real individual posts with their
  own dates, text, likes and author follower counts. I used to crawl
  whole discussion pages (LIHKG / Naver / Daum / Yahoo Community) —
  a page is not a post, so that gave 0 mentions and 0.00 sentiment,
  and I dropped it on 10 Nov 2026. (Telegram was dropped too — there
  is no official Apify actor called "apify/telegram-scraper".)
* Prices: real daily prices from Yahoo Finance (free, no key).
* Sentiment: VADER — a word list, no API, no paid model.
* Model: Ridge regression (scikit-learn), tested on a time split
  (train first 70% of days, test last 30%).

Be honest about the data (read this before judging any number):
* PRICES are REAL — daily closes from Yahoo Finance, fetched live.
* SOCIAL POSTS / NEWS are SAMPLE unless Apify/Exa tokens are set.
  With no tokens it runs in SAMPLE mode: it scores a small set of
  clearly-labelled example posts (src/sample.py) so everything still
  runs end to end. Sample posts are NEVER passed off as scraped posts —
  every output file and the report carry a "SAMPLE SENTIMENT" note
  in sample mode.
* With free Apify + Exa tokens in .env the collectors switch to LIVE
  mode and pull real posts/news instead. The Stocktwits API needs no
  token, so even without an Apify token the run is still live for the
  tickers that have a Stocktwits stream. Tokens are never written
  in the code and never committed.

By Monu Mishra (BCA 2026) — Python, scikit-learn, NLP, LangChain/RAG.
"""
