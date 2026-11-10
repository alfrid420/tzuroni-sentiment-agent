"""Tickers, exchanges and sentiment sources for the TZURONI task.

I checked all 10 tickers on 10 Nov 2026 — they are real, liquid stocks:
HK: 0700 Tencent, 9988 Alibaba, 3690 Meituan, 9618 JD.com, 1810 Xiaomi.
KR:  005930 Samsung Electronics, 000660 SK Hynix, 035420 Naver,
      005380 Hyundai Motor, 051910 LG Chem.
Yahoo symbols: "<code>.HK" for Hong Kong, "<code>.KS" for Korea.
"""

TICKERS = [
    # --- Hong Kong (HKEX, currency HKD) ---
    {"ticker": "0700", "name": "Tencent Holdings", "exchange": "Hong Kong",
     "yahoo": "0700.HK", "currency": "HKD",
     "keywords": ["Tencent", "$0700", "0700.HK", "WeChat", "Tencent Games"]},
    {"ticker": "9988", "name": "Alibaba Group", "exchange": "Hong Kong",
     "yahoo": "9988.HK", "currency": "HKD",
     "keywords": ["Alibaba", "$9988", "9988.HK", "Alibaba Cloud", "Taobao"]},
    {"ticker": "3690", "name": "Meituan", "exchange": "Hong Kong",
     "yahoo": "3690.HK", "currency": "HKD",
     "keywords": ["Meituan", "$3690", "3690.HK", "Meituan food delivery"]},
    {"ticker": "9618", "name": "JD.com", "exchange": "Hong Kong",
     "yahoo": "9618.HK", "currency": "HKD",
     "keywords": ["JD.com", "$9618", "9618.HK", "JD Logistics"]},
    {"ticker": "1810", "name": "Xiaomi", "exchange": "Hong Kong",
     "yahoo": "1810.HK", "currency": "HKD",
     "keywords": ["Xiaomi", "$1810", "1810.HK", "Xiaomi phone", "Xiaomi EV"]},
    # --- South Korea (KRX, currency KRW) ---
    {"ticker": "005930", "name": "Samsung Electronics", "exchange": "South Korea",
     "yahoo": "005930.KS", "currency": "KRW",
     "keywords": ["Samsung Electronics", "005930", "Samsung chip", "Galaxy"]},
    {"ticker": "000660", "name": "SK Hynix", "exchange": "South Korea",
     "yahoo": "000660.KS", "currency": "KRW",
     "keywords": ["SK Hynix", "000660", "Hynix HBM", "SK Hynix memory"]},
    {"ticker": "035420", "name": "Naver", "exchange": "South Korea",
     "yahoo": "035420.KS", "currency": "KRW",
     "keywords": ["Naver", "035420", "Naver AI", "Naver search"]},
    {"ticker": "005380", "name": "Hyundai Motor", "exchange": "South Korea",
     "yahoo": "005380.KS", "currency": "KRW",
     "keywords": ["Hyundai Motor", "005380", "Hyundai EV", "Hyundai sales"]},
    {"ticker": "051910", "name": "LG Chem", "exchange": "South Korea",
     "yahoo": "051910.KS", "currency": "KRW",
     "keywords": ["LG Chem", "051910", "LG Chem battery", "LG Chem materials"]},
]

# Sentiment sources per exchange (from the assignment doc — min. 2 each).
# Both give me real individual posts with their own timestamps now:
#  1. Stocktwits public API (free, no key) — per-message stream per
#     US ADR symbol. I checked each symbol's title on 10 Nov 2026, so
#     I know the stream is really about the company.
#  2. X (Twitter) search via the Apify actor apisamurai/x-twitter-scraper
#     (pay-per-tweet, $0.25 per 1,000; about $0.08 for a full run).
# What I dropped, and why:
#  * Apify website-content-crawler on LIHKG / Naver / Daum / Yahoo
#    discussion pages — a page crawl returns page text, not individual
#    dated posts, so likes/reposts/views/followers were all 0 and the
#    posts never lined up with price days. That was the gap.
#  * Telegram — there is no official Apify actor called
#    "apify/telegram-scraper", I checked, so I did not use a made-up ID.
#  * Reddit — its public JSON endpoints answer 403 without a login now,
#    so I could not get posts from there for free.
#  * Naver / Daum / LIHKG boards — no free per-post actor or API I could
#    verify; the page crawls came back nearly empty anyway.
SOURCES = {
    "Hong Kong": [
        {"name": "Stocktwits", "kind": "social",
         "note": "Per-message stream for the US ADR symbol (public API)"},
        {"name": "X (Twitter)", "kind": "social",
         "note": "Keyword search via Apify actor apisamurai/x-twitter-scraper"},
    ],
    "South Korea": [
        {"name": "Stocktwits", "kind": "social",
         "note": "Per-message stream for the US ADR symbol (public API) — only Samsung and Hyundai have one"},
        {"name": "X (Twitter)", "kind": "social",
         "note": "Keyword search via Apify actor apisamurai/x-twitter-scraper"},
    ],
}

# Apify actors used in LIVE mode (free-tier actors; IDs are public, not secrets).
# Only one actor now: apisamurai/x-twitter-scraper (checked — it exists,
# public, 1,387 runs, $0.25 per 1,000 tweets). The token itself comes
# from the APIFY_TOKEN environment variable only.
APIFY_ACTORS = {
    "x": "apisamurai/x-twitter-scraper",  # X search, per-tweet rows
}

# Stocktwits symbol per ticker (US ADR). I verified each one by reading
# the stream's own "symbol.title" on 10 Nov 2026 — bare numbers like
# "005930" are a trap: Stocktwits reads them as other instruments
# ("005930" is the IWM ETF, "000660" is Treasury futures), so I only
# use a symbol when its title names the company. None means Stocktwits
# has no stream for that company at all.
STOCKTWITS_SYMBOLS = {
    "0700": "TCEHY",    # Tencent Holdings Ltd. - ADR
    "9988": "BABA",     # Alibaba
    "3690": "MPNGY",    # Meituan Unsponsored ADR (real, but quiet: newest post 8 Sep 2026)
    "9618": "JD",       # JD.com
    "1810": "XIACY",    # Xiaomi
    "005930": "SSNLF",  # Samsung Electronics
    "000660": None,     # SK Hynix — no Stocktwits symbol exists
    "035420": None,     # Naver — no Stocktwits symbol exists
    "005380": "HYMTF",  # Hyundai Motor Company GDR
    "051910": None,     # LG Chem — no Stocktwits symbol exists
}

# X search term per ticker (plain company name + "stock" where it helps
# cut noise; lang:en and -filter:replies are added by the collector).
X_SEARCH_TERMS = {
    "0700": "Tencent stock",
    "9988": "Alibaba stock",
    "3690": "Meituan",
    "9618": "JD.com stock",
    "1810": "Xiaomi stock",
    "005930": "Samsung Electronics stock",
    "000660": "SK Hynix",
    "035420": "Naver stock",
    "005380": "Hyundai Motor stock",
    "051910": "LG Chem stock",
}

# Exa search: news only, recent, per ticker.
EXA_NEWS_PER_TICKER = 5

# Pipeline windows
PRICE_HISTORY_DAYS = 365     # price bars fetched (~250 trading days: more
                             # training days than the old 90-day window)
SENTIMENT_WINDOW_DAYS = 10   # social posts window required by the task
SUPPORT_RESISTANCE_WINDOW = 20  # rolling window for S/R levels
