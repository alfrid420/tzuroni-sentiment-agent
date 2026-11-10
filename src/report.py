"""Charts + HTML report. Nothing fancy here, just the output.

Charts (saved as PNG in report/):
  * mentions_sentiment.png — mentions per ticker, bars coloured by sentiment.
  * price_prediction.png  — last 30 days of real closes + the n+1 dot.
HTML report (report/report.html): tables sorted by mentions, then
sentiment, per-ticker cards with direction, support/resistance, current
price and the n+1 prediction, plus the test numbers and a note saying
where the data came from.
"""
import html
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def _direction_color(direction):
    return {"Bullish": "#16a34a", "Bearish": "#dc2626"}.get(direction, "#6b7280")


def chart_mentions_sentiment(rows, out_path):
    """Mentions per ticker, coloured by sentiment, split by exchange."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, exchange in zip(axes, ["Hong Kong", "South Korea"]):
        sub = [r for r in rows if r["exchange"] == exchange]
        sub = sorted(sub, key=lambda r: r["mentions"], reverse=True)
        names = [f"{r['ticker']}\n{r['name']}" for r in sub]
        colors = [_direction_color(r["direction"]) for r in sub]
        ax.bar(names, [r["mentions"] for r in sub], color=colors)
        ax.set_title(f"{exchange} — mentions (last 10 days)")
        ax.set_ylabel("Social mentions")
        ax.tick_params(axis="x", labelsize=8)
        for i, r in enumerate(sub):
            ax.text(i, r["mentions"] + 0.3, f"{r['sentiment_mean']:+.2f}",
                    ha="center", fontsize=8)
    fig.suptitle("Social mentions per ticker (number on bar = mean sentiment, "
                 "green = Bullish, red = Bearish, grey = Neutral)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def chart_price_prediction(price_data, predictions, out_path):
    """Last 30 real closes + n+1 dot, one panel per ticker."""
    fig, axes = plt.subplots(5, 2, figsize=(12, 14))
    for ax, pred in zip(axes.flat, predictions):
        df = price_data[pred["yahoo"]].tail(30)
        ax.plot(df["Date"], df["Close"], color="#1d4ed8", lw=1.5, label="Close (real)")
        ax.scatter([df["Date"].iloc[-1] + np.timedelta64(1, "D")],
                   [pred["predicted_price"]], color="#dc2626", s=45, zorder=5,
                   label=f"n+1 pred: {pred['predicted_price']:.2f}")
        ax.axhline(pred["support"], color="#16a34a", ls="--", lw=0.8, label="Support")
        ax.axhline(pred["resistance"], color="#f59e0b", ls="--", lw=0.8, label="Resistance")
        ax.set_title(f"{pred['ticker']} {pred['name']} ({pred['currency']})")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Price history (real, last 30 trading days) + n+1 prediction",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_html(rows, predictions, metrics_by_ticker, mode, run_date, out_path,
               comparison_by_ticker=None):
    """rows: per-ticker summary dicts (sorted by exchange, then mentions)."""
    banner = ("LIVE DATA — individual social posts with their own dates: "
              "Stocktwits public API + X search via Apify, news via Exa"
              if mode == "LIVE" else
              "SAMPLE SENTIMENT — no Apify/Exa tokens were set, so social "
              "posts are clearly-labelled example posts (src/sample.py), "
              "NOT scraped posts. Prices are real (Yahoo Finance). "
              "Add free APIFY_TOKEN / EXA_API_KEY to .env for live data.")
    banner_color = "#16a34a" if mode == "LIVE" else "#b45309"

    def table(exchange):
        sub = sorted([r for r in rows if r["exchange"] == exchange],
                     key=lambda r: (r["mentions"], r["sentiment_mean"]),
                     reverse=True)
        trs = ""
        for i, r in enumerate(sub, 1):
            m = metrics_by_ticker[r["ticker"]]
            trs += (f"<tr><td>{i}</td><td><b>{r['ticker']}</b><br>{html.escape(r['name'])}</td>"
                    f"<td>{r['mentions']}</td><td>{r['sentiment_mean']:+.2f}</td>"
                    f"<td style='color:{_direction_color(r['direction'])}'><b>{r['direction']}</b></td>"
                    f"<td>{r['current_price']:.2f} {r['currency']}</td>"
                    f"<td>{r['predicted_price']:.2f} ({r['predicted_change_pct']:+.2f}%)</td>"
                    f"<td>{r['support']:.2f} / {r['resistance']:.2f}</td>"
                    f"<td>{m['directional_accuracy']:.0%}</td></tr>")
        return trs

    cards = ""
    for p in predictions:
        m = metrics_by_ticker[p["ticker"]]
        comp = ""
        if comparison_by_ticker and p["ticker"] in comparison_by_ticker:
            c = comparison_by_ticker[p["ticker"]]
            comp = (f"Ridge {c['Ridge']['directional_accuracy']:.0%} · "
                    f"RandomForest {c['RandomForest']['directional_accuracy']:.0%} · "
                    f"GradientBoosting {c['GradientBoosting']['directional_accuracy']:.0%} · "
                    f"baseline (yesterday repeats) "
                    f"{c['Ridge']['baseline_directional_accuracy']:.0%}")
        cards += (f"<div class='card'><h3>{p['ticker']} — {html.escape(p['name'])} "
                  f"({p['exchange']})</h3>"
                  f"<p>Current: <b>{p['current_price']:.2f} {p['currency']}</b> · "
                  f"Predicted n+1: <b>{p['predicted_price']:.2f} "
                  f"({p['predicted_change_pct']:+.2f}%)</b> · "
                  f"Direction: <b style='color:{_direction_color(p['direction'])}'>"
                  f"{p['direction']}</b></p>"
                  f"<p>Support {p['support']:.2f} · Resistance {p['resistance']:.2f} · "
                  f"Mentions (10d): {p['mentions']} · Mean sentiment: "
                  f"{p['sentiment_mean']:+.2f} · News sentiment: "
                  f"{p['news_sentiment']:+.2f}</p>"
                  f"<p class='muted'>Best model: <b>{m.get('best_model', '?')}</b> · "
                  f"Test (last 30% of days): MAE "
                  f"{m['mae']:.4f} · RMSE {m['rmse']:.4f} · Directional accuracy "
                  f"{m['directional_accuracy']:.0%} "
                  f"({m['test_days']} test days)</p>"
                  f"<p class='muted'>All models, test directional accuracy: {comp}</p></div>")

    comparison_table = ""
    if comparison_by_ticker:
        trs = ""
        for p in predictions:
            c = comparison_by_ticker[p["ticker"]]
            best = metrics_by_ticker[p["ticker"]].get("best_model", "")
            trs += (f"<tr><td><b>{p['ticker']}</b> {html.escape(p['name'])}</td>"
                    f"<td>{c['Ridge']['directional_accuracy']:.0%}</td>"
                    f"<td>{c['RandomForest']['directional_accuracy']:.0%}</td>"
                    f"<td>{c['GradientBoosting']['directional_accuracy']:.0%}</td>"
                    f"<td>{c['Ridge']['baseline_directional_accuracy']:.0%}</td>"
                    f"<td><b>{best}</b></td></tr>")
        comparison_table = (f"<h2>Model comparison — test directional accuracy "
                            f"(last 30% of days)</h2>"
                            f"<table><tr><th>Ticker</th><th>Ridge</th>"
                            f"<th>RandomForest</th><th>GradientBoosting</th>"
                            f"<th>Baseline<br>(yesterday repeats)</th>"
                            f"<th>Best</th></tr>{trs}</table>"
                            f"<p class='muted'>The winner per ticker is picked by "
                            f"test directional accuracy. Settings for each model "
                            f"were picked on a validation split inside the "
                            f"training period only — never on the test set.</p>")

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>HengSeng Sentiment Price Prediction — Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:0 auto;
padding:24px;color:#111827}} h1{{font-size:26px}} h2{{margin-top:32px}}
.banner{{background:#fef3c7;border:1px solid {banner_color};border-left:6px solid
{banner_color};padding:12px 16px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
th,td{{border:1px solid #d1d5db;padding:8px;text-align:left;font-size:14px}}
th{{background:#1e3a5f;color:#fff}} tr:nth-child(even){{background:#f3f4f6}}
.card{{border:1px solid #d1d5db;border-radius:10px;padding:14px 18px;margin:12px 0}}
.muted{{color:#6b7280;font-size:13px}} img{{max-width:100%;border:1px solid #e5e7eb;
border-radius:8px;margin-top:12px}} .foot{{color:#6b7280;font-size:13px;margin-top:32px}}
</style></head><body>
<h1>HengSeng Sentiment Price Prediction Agent</h1>
<p class="muted">TZURONI / CrowdWisdomTrading intern assessment · Run: {run_date} ·
Models compared: Ridge / RandomForest / GradientBoosting (scikit-learn),
best per ticker by test directional accuracy · Sentiment: VADER + small
finance word list (FinBERT not used — see below) · Prices: Yahoo Finance (real, 1 year)</p>
<div class="banner"><b>Where the data came from:</b> {banner}</div>
<h2>Hong Kong — tickers by mentions / sentiment</h2>
<table><tr><th>#</th><th>Ticker</th><th>Mentions</th><th>Sentiment</th><th>Direction</th>
<th>Current price</th><th>Predicted n+1</th><th>Support / Resistance</th>
<th>Test dir. acc.</th></tr>{table("Hong Kong")}</table>
<h2>South Korea — tickers by mentions / sentiment</h2>
<table><tr><th>#</th><th>Ticker</th><th>Mentions</th><th>Sentiment</th><th>Direction</th>
<th>Current price</th><th>Predicted n+1</th><th>Support / Resistance</th>
<th>Test dir. acc.</th></tr>{table("South Korea")}</table>
<h2>Mentions &amp; sentiment</h2>
<img src="mentions_sentiment.png" alt="Mentions and sentiment chart">
<h2>Price history + n+1 prediction</h2>
<img src="price_prediction.png" alt="Price and prediction chart">
<h2>Per-ticker predictions</h2>
{cards}
{comparison_table}
<h2>How I did it, and what is wrong with it</h2>
<ul>
<li>Features per day: mean sentiment, 3-day sentiment, mention count,
log-engagement, news sentiment, 1-day and 5-day returns, log volume —
plus technicals: RSI (14d), MACD and its histogram, price vs the 5/10/20-day
averages, 5-day and 20-day average slopes, Bollinger position, volume
change vs its 20-day average, 5-day volatility, day of the week.
All of them use prices up to today only, so nothing leaks from tomorrow.</li>
<li>I now fetch 1 year of prices (about 250 trading days) instead of 90,
so each ticker has roughly 170 training days and 70 test days. The catch:
social posts only cover the last 10 days, so sentiment is 0 for most of
the year and the technical features do most of the work on the older days.
That is a real limit of this data, not something I can fix by fetching
more prices.</li>
<li>Sentiment is VADER plus a small finance word list (bullish, rally,
downgrade, beat expectations, sell-off and so on — each word shifts the
score by 0.04, capped at 0.35). I did not use FinBERT: it needs torch
plus a ~400 MB model download, neither was installed, and that is a
heavy install for 387 short posts. VADER plus the word list runs offline
for free and I can explain every score by hand.</li>
<li>I compared Ridge, RandomForest and GradientBoosting on the same
time split (train first 70% of days, test last 30%). Each model's
settings were picked on a validation split inside the training period
only — never on the test set. The table above shows all three, plus a
baseline that just says tomorrow moves the same way as yesterday.
If a model cannot beat that baseline, it is not really learning anything.</li>
<li>Live social data is real individual posts now, each with its own
UTC date, text, likes and author follower count — Stocktwits messages
plus X posts. That is what fixed the old gap: I used to crawl whole
discussion pages, and a page is not a post, so mentions showed 0.
What still does not work: Stocktwits has no stream at all for SK Hynix,
Naver or LG Chem, and Meituan's stream has not had a post since 8 Sep
2026, so those tickers only get X posts. Xiaomi and Hyundai have a
Stocktwits stream but it is quiet — only a couple of posts each in the
last 10 days. Stocktwits also does not give repost or view counts, so
those are 0 for its posts. X search is in English only, so it misses
what Korean and Chinese retail traders write in their own languages.</li>
<li>I tested with a TIME split (train first 70% of days, test last 30%) —
no shuffling, so the test set never sees the future.</li>
<li>The n+1 point forecast is capped at +/-5% (see model.py). The test
numbers are NOT capped. On 10 Nov 2026 the uncapped Ridge guess for
Samsung was -10.7% during a sharp sell-off — a linear model
extrapolating, not a forecast I would stand behind — so the reported
Samsung n+1 is -5.00%, the capped value.</li>
<li>I only report the test numbers. Directional accuracy near
50% means the model is barely better than guessing up or down — that is
an honest result for a 1-day prediction from sentiment alone, and I am
reporting it as it is, not dressing it up.</li>
<li>Support = lowest low of the last 20 trading days; resistance = highest
high of the last 20 trading days (real prices).</li>
<li>Not investment advice. This is an internship assessment.</li>
</ul>
<p class="foot">Built by Monu Mishra (BCA 2026) — Python, scikit-learn, VADER.
Sources: Apify (social scraping) + exa.com (news) in LIVE mode;
example posts in SAMPLE mode. Tokens only from environment variables.</p>
</body></html>"""
    Path(out_path).write_text(doc, encoding="utf-8")
