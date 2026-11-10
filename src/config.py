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
# I collect them with Apify's website-content-crawler (see below).
# Note: there is no official Apify actor called "apify/telegram-scraper"
# — I checked, the page does not exist — so I dropped Telegram as a
# source instead of using a made-up actor ID. The sources below are the
# ones the crawler actually visits.
SOURCES = {
    "Hong Kong": [
        {"name": "LIHKG", "kind": "forum",
         "note": "HK retail-trader forum, stock search page"},
        {"name": "Stocktwits", "kind": "social",
         "note": "Stock discussion stream per ticker"},
        {"name": "Yahoo Finance Community", "kind": "forum",
         "note": "Yahoo conversation page per ticker"},
    ],
    "South Korea": [
        {"name": "Naver Finance Boards", "kind": "forum",
         "note": "KRX stock discussion rooms (jongmok-toronsil)"},
        {"name": "Daum Finance Forums", "kind": "forum",
         "note": "Korean financial portal discussion page per ticker"},
        {"name": "Yahoo Finance Community", "kind": "forum",
         "note": "Yahoo conversation page per ticker"},
    ],
}

# Apify actors used in LIVE mode (free-tier actors; IDs are public, not secrets).
# Only one actor now: apify/website-content-crawler (checked — it exists,
# Apify's own actor, 4.6 stars, 170k+ users). The token itself comes from
# the APIFY_TOKEN environment variable only.
APIFY_ACTORS = {
    "web": "apify/website-content-crawler",  # forum / discussion pages
}


def discussion_urls(t):
    """Discussion pages to crawl for one ticker. Gives back a list of
    (source name, url). 3 pages per ticker, 2+ sources per exchange."""
    from urllib.parse import quote_plus

    if t["exchange"] == "Hong Kong":
        return [
            ("LIHKG",
             f"https://lihkg.com/search?q={quote_plus(t['name'])}"),
            ("Stocktwits",
             f"https://stocktwits.com/symbol/{t['ticker']}.HK"),
            ("Yahoo Finance Community",
             f"https://finance.yahoo.com/quote/{t['ticker']}.HK/community"),
        ]
    return [
        ("Naver Finance Boards",
         f"https://finance.naver.com/item/board.naver?code={t['ticker']}"),
        ("Daum Finance Forums",
         f"https://finance.daum.net/quotes/A{t['ticker']}/discussion"),
        ("Yahoo Finance Community",
         f"https://finance.yahoo.com/quote/{t['ticker']}.KS/community"),
    ]

# Exa search: news only, recent, per ticker.
EXA_NEWS_PER_TICKER = 5

# Pipeline windows
PRICE_HISTORY_DAYS = 90      # price bars fetched (enough for train/test + S/R)
SENTIMENT_WINDOW_DAYS = 10   # social posts window required by the task
SUPPORT_RESISTANCE_WINDOW = 20  # rolling window for S/R levels
