"""Features, support/resistance, model and testing.

Features per ticker per day (all from real prices + scored sentiment):
  * sentiment_mean   — mean VADER score of that day's posts (0 if none)
  * sentiment_3d     — 3-day average sentiment
  * mention_count    — posts that day
  * engagement       — likes + reposts + views that day (log1p)
  * news_sentiment   — mean VADER score of that day's news (0 if none)
  * return_1d        — yesterday's price return
  * return_5d        — 5-day price return
  * volume_log       — log trading volume

Target: tomorrow's return (price at n+1 vs price at n).
Model: Ridge regression (scikit-learn) — simple, and I can see what it does.
Testing: TIME split — train on the first 70% of days, test on the
last 30%. No shuffling (shuffling would let the test set peek at the
future). I only report the test numbers: MAE, RMSE, direction accuracy
(did I get up/down right?). The n+1 prediction uses the model trained
on all data up to today, on today's features.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["sentiment_mean", "sentiment_3d", "mention_count",
            "engagement", "news_sentiment", "return_1d", "return_5d",
            "volume_log"]


def support_resistance(df, window=20):
    """Support = lowest low, resistance = highest high, last 20 days."""
    closes = df["Close"]
    return {
        "support": float(df["Low"].rolling(window).min().iloc[-1]),
        "resistance": float(df["High"].rolling(window).max().iloc[-1]),
        "current_price": float(closes.iloc[-1]),
        "prev_close": float(closes.iloc[-2]) if len(closes) > 1 else float(closes.iloc[-1]),
    }


def build_daily_features(price_df, posts, news):
    """Put real prices and scored sentiment together, one row per day."""
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["volume_log"] = np.log1p(df["Volume"].fillna(0))

    # daily sentiment from posts (UTC dates)
    post_rows = []
    for p in posts:
        ts = p.get("timestamp_utc", "")
        try:
            d = pd.to_datetime(ts).date()
        except Exception:  # noqa: BLE001
            continue
        post_rows.append({"date": d, "sentiment": p["sentiment"],
                          "engagement": (p.get("likes", 0) or 0)
                          + (p.get("reposts", 0) or 0)
                          + (p.get("views", 0) or 0)})
    if post_rows:
        pf = pd.DataFrame(post_rows)
        agg = pf.groupby("date").agg(
            sentiment_mean=("sentiment", "mean"),
            mention_count=("sentiment", "size"),
            engagement=("engagement", "sum")).reset_index()
        df = df.merge(agg, on="date", how="left")
    if "sentiment_mean" not in df:
        df["sentiment_mean"] = 0.0
    else:
        df["sentiment_mean"] = df["sentiment_mean"].fillna(0.0)
    if "mention_count" not in df:
        df["mention_count"] = 0
    else:
        df["mention_count"] = df["mention_count"].fillna(0)
    if "engagement" not in df:
        df["engagement"] = 0.0
    else:
        df["engagement"] = df["engagement"].fillna(0)
    df["engagement"] = np.log1p(df["engagement"])
    df["sentiment_3d"] = df["sentiment_mean"].rolling(3, min_periods=1).mean()

    news_rows = []
    for n in news:
        try:
            d = pd.to_datetime(n.get("published", "")).date()
        except Exception:  # noqa: BLE001
            continue
        news_rows.append({"date": d, "sentiment": n["sentiment"]})
    if news_rows:
        nf = pd.DataFrame(news_rows)
        na = nf.groupby("date").agg(news_sentiment=("sentiment", "mean")).reset_index()
        df = df.merge(na, on="date", how="left")
    if "news_sentiment" not in df:
        df["news_sentiment"] = 0.0
    else:
        df["news_sentiment"] = df["news_sentiment"].fillna(0.0)

    df["target_return"] = df["Close"].shift(-1) / df["Close"] - 1  # n+1
    return df


def train_and_evaluate(df):
    """Time-split test. Gives back (model, metrics, test_df)."""
    work = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)
    if len(work) < 10:
        raise RuntimeError("Not enough days with features to train (need >= 10)")
    split = int(len(work) * 0.7)
    train, test = work.iloc[:split], work.iloc[split:]
    model = Ridge(alpha=1.0)
    model.fit(train[FEATURES], train["target_return"])
    pred = model.predict(test[FEATURES])
    actual = test["target_return"].to_numpy()
    direction_ok = ((np.sign(pred) == np.sign(actual)) |
                    ((pred == 0) & (actual == 0))).mean()
    metrics = {
        "train_days": len(train), "test_days": len(test),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(float(np.sqrt(mean_squared_error(actual, pred)))),
        "directional_accuracy": float(direction_ok),
    }
    return model, metrics, test.assign(pred_return=pred)


def predict_next_day(model, df):
    """Guess tomorrow's return from today's features."""
    last = df.dropna(subset=FEATURES).iloc[-1]
    features = last[FEATURES].to_frame().T  # keep feature names, no warning
    pred_return = float(model.predict(features)[0])
    current = float(last["Close"])
    return pred_return, current * (1 + pred_return)
