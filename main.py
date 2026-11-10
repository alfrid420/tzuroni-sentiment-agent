"""Main pipeline: collect -> score -> features -> model -> report.

Run:
    python main.py            (SAMPLE mode if no tokens are set)
    APIFY_TOKEN=... EXA_API_KEY=... python main.py   (LIVE mode)

What it writes (all inside this project folder):
    data/raw_prices.csv        real daily prices, all tickers
    data/posts_scored.csv      scored social posts (with data_origin)
    data/news_scored.csv       scored news (LIVE mode; empty in SAMPLE mode)
    data/daily_features.csv    daily features + n+1 target, all tickers
    data/predictions.csv       n+1 predictions + support/resistance + test numbers
    report/report.html         the visual report (open it in a browser)
    report/mentions_sentiment.png
    report/price_prediction.png
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import collectors, model as model_mod, report as report_mod  # noqa: E402
from src.config import (PRICE_HISTORY_DAYS, SENTIMENT_WINDOW_DAYS,  # noqa: E402
                        SUPPORT_RESISTANCE_WINDOW, TICKERS)
from src.sentiment import direction_from_sentiment  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORT = ROOT / "report"


def main():
    load_dotenv()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"== HengSeng Sentiment Price Prediction Agent — run {run_date} ==")

    # 1. get real prices first
    print("[1/5] Fetching real prices (Yahoo Finance)...")
    prices = collectors.fetch_prices(TICKERS, days=PRICE_HISTORY_DAYS)
    price_rows = []
    for t in TICKERS:
        df = prices[t["yahoo"]].copy()
        df["ticker"] = t["ticker"]
        price_rows.append(df)
    pd.concat(price_rows).to_csv(DATA / "raw_prices.csv", index=False)

    # 2. social posts (LIVE with Apify, else SAMPLE) + news (LIVE with Exa)
    print("[2/5] Collecting social posts + news...")
    posts, social_mode = collectors.collect_social(TICKERS, days=SENTIMENT_WINDOW_DAYS)
    news, news_mode = collectors.collect_news(TICKERS)
    mode = "LIVE" if social_mode == "LIVE" else "SAMPLE"
    print(f"      social mode: {social_mode} ({len(posts)} posts) · "
          f"news mode: {news_mode} ({len(news)} items)")
    scored_posts = collectors.score_posts(posts)
    scored_news = collectors.score_news(news)
    pd.DataFrame(scored_posts).to_csv(DATA / "posts_scored.csv", index=False)
    pd.DataFrame(scored_news).to_csv(DATA / "news_scored.csv", index=False)

    # 3. features + model + tomorrow's prediction, one ticker at a time
    print("[3/5] Building features, training Ridge, testing on the last 30%...")
    rows, predictions, metrics_by_ticker = [], [], {}
    feature_frames = []
    for t in TICKERS:
        t_posts = [p for p in scored_posts if p["ticker"] == t["ticker"]]
        t_news = [n for n in scored_news if n["ticker"] == t["ticker"]]
        df = model_mod.build_daily_features(prices[t["yahoo"]], t_posts, t_news)
        df["ticker"] = t["ticker"]
        feature_frames.append(df)
        model, metrics, _test = model_mod.train_and_evaluate(df)
        pred_return, pred_price = model_mod.predict_next_day(model, df)
        sr = model_mod.support_resistance(df, window=SUPPORT_RESISTANCE_WINDOW)
        mentions = int(df["mention_count"].tail(SENTIMENT_WINDOW_DAYS).sum())
        sent_mean = float(df["sentiment_mean"].tail(SENTIMENT_WINDOW_DAYS).mean())
        news_sent = float(df["news_sentiment"].tail(SENTIMENT_WINDOW_DAYS).mean())
        direction = direction_from_sentiment(sent_mean)
        row = {
            "ticker": t["ticker"], "name": t["name"], "exchange": t["exchange"],
            "yahoo": t["yahoo"], "currency": t["currency"],
            "mentions": mentions, "sentiment_mean": sent_mean,
            "news_sentiment": news_sent, "direction": direction,
            "current_price": sr["current_price"],
            "predicted_price": pred_price,
            "predicted_change_pct": (pred_price / sr["current_price"] - 1) * 100,
            "support": sr["support"], "resistance": sr["resistance"],
            "position": "No position held (assessment — no trading)",
        }
        rows.append(row)
        predictions.append(row)
        metrics_by_ticker[t["ticker"]] = metrics
        print(f"      {t['ticker']:>7} {t['name']:<22} {direction:<8} "
              f"now {sr['current_price']:>10.2f} -> n+1 {pred_price:>10.2f} "
              f"({row['predicted_change_pct']:+.2f}%)  test dir-acc "
              f"{metrics['directional_accuracy']:.0%}  MAE {metrics['mae']:.4f}")

    pd.concat(feature_frames).to_csv(DATA / "daily_features.csv", index=False)
    pred_df = pd.DataFrame([{
        **{k: v for k, v in r.items() if k != "yahoo"},
        "mae": metrics_by_ticker[r["ticker"]]["mae"],
        "rmse": metrics_by_ticker[r["ticker"]]["rmse"],
        "directional_accuracy": metrics_by_ticker[r["ticker"]]["directional_accuracy"],
        "data_mode": mode,
    } for r in predictions])
    pred_df.to_csv(DATA / "predictions.csv", index=False)

    # 4. charts + HTML report
    print("[4/5] Building visual report...")
    report_mod.chart_mentions_sentiment(rows, REPORT / "mentions_sentiment.png")
    report_mod.chart_price_prediction(prices, predictions, REPORT / "price_prediction.png")
    report_mod.build_html(rows, predictions, metrics_by_ticker, mode, run_date,
                          REPORT / "report.html")

    # 5. print a short summary
    print("[5/5] Done.")
    print(f"      mode: {mode} (set APIFY_TOKEN + EXA_API_KEY in .env for LIVE)")
    print(f"      report: {REPORT / 'report.html'}")
    print(f"      predictions: {DATA / 'predictions.csv'}")
    if mode == "SAMPLE":
        print("      NOTE: sentiment is SAMPLE — add free tokens to .env "
              "and re-run for live scraped posts/news.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
