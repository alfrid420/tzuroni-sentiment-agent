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


def build_html(rows, predictions, metrics_by_ticker, mode, run_date, out_path):
    """rows: per-ticker summary dicts (sorted by exchange, then mentions)."""
    banner = ("LIVE DATA — social posts scraped via Apify, news via Exa"
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
                  f"<p class='muted'>Test (last 30% of days): MAE "
                  f"{m['mae']:.4f} · RMSE {m['rmse']:.4f} · Directional accuracy "
                  f"{m['directional_accuracy']:.0%} "
                  f"({m['test_days']} test days)</p></div>")

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
Model: Ridge regression (scikit-learn) · Sentiment: VADER · Prices: Yahoo Finance (real)</p>
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
<h2>How I did it, and what is wrong with it</h2>
<ul>
<li>Features per day: mean sentiment, 3-day sentiment, mention count,
log-engagement, news sentiment, 1-day and 5-day returns, log volume.</li>
<li>Live social data has one real limit I should say out loud: each
crawled discussion page is one record, dated at crawl time. Pages do
not give me individual posts with their own dates, so the live posts
do not line up with past price days — that is why mentions show 0 in
the table above. The live news (50 dated items from Exa) does line up
with price days and does feed the model. LIHKG search pages and Daum
discussion pages came back nearly empty, so they gave nothing;
Stocktwits, Yahoo Community and Naver boards gave the 19 live pages.</li>
<li>I tested with a TIME split (train first 70% of days, test last 30%) —
no shuffling, so the test set never sees the future.</li>
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
