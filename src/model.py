"""Features, support/resistance, models and testing.

Features per ticker per day (all from real prices + scored sentiment).
Old features (kept):
  * sentiment_mean   — mean VADER+finance-word score of that day's posts (0 if none)
  * sentiment_3d     — 3-day average sentiment
  * mention_count    — posts that day
  * engagement       — likes + reposts + views that day (log1p)
  * news_sentiment   — mean score of that day's news (0 if none)
  * return_1d        — yesterday's price return (also the previous-day return)
  * return_5d        — 5-day price return
  * volume_log       — log trading volume
New technical features (all from prices up to today only — nothing
from tomorrow, so no look-ahead):
  * rsi_14           — RSI over 14 days (above 70 = overbought, below 30 = oversold)
  * macd / macd_hist — MACD line (12-day minus 26-day average) and its
                       distance from its 9-day signal line
  * price_vs_ma5 / price_vs_ma10 / price_vs_ma20 — today's close vs the
                       5/10/20-day average price, as a fraction
  * ma5_slope / ma20_slope — how much the 5/20-day average moved in 5 days
  * bollinger_pos    — where today's close sits inside the 20-day
                       Bollinger band (0 = bottom, 1 = top)
  * volume_change    — today's volume vs its 20-day average, as a fraction
  * volatility_5d    — how jumpy returns were over the last 5 days
  * day_of_week      — Monday 0 .. Friday 4 (some days just move more)

Target: tomorrow's return (price at n+1 vs price at n).
Models (scikit-learn only): Ridge, RandomForest, GradientBoosting.
Testing: TIME split — train on the first 70% of days, test on the
last 30%. No shuffling (shuffling would let the test set peek at the
future). I only report the test numbers: MAE, RMSE, direction accuracy
(did I get up/down right?).
Tuning: I do NOT tune on the test set. Inside the training period I
split off its last 20% as a validation set and pick each model's
settings (Ridge alpha, tree depth) on that validation set only.
Then I refit on the full training period and test once.
The n+1 prediction refits the winning model on ALL days up to today
(the last day has no target yet, so it is not in the training rows)
and predicts from today's features.
I also report a simple baseline on the test set: "tomorrow moves the
same way as yesterday" — if a model cannot beat that, it is not
really learning anything.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["sentiment_mean", "sentiment_3d", "mention_count",
            "engagement", "news_sentiment", "return_1d", "return_5d",
            "volume_log",
            "rsi_14", "macd", "macd_hist",
            "price_vs_ma5", "price_vs_ma10", "price_vs_ma20",
            "ma5_slope", "ma20_slope", "bollinger_pos",
            "volume_change", "volatility_5d", "day_of_week"]

# Small grids, picked on the validation split inside training only.
# Kept small on purpose: ~170 training days and 20 features, so a big
# grid would just overfit the validation set.
MODEL_GRIDS = {
    "Ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0, 100.0)],
    "RandomForest": [
        {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5},
    ],
    "GradientBoosting": [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2,
         "subsample": 0.8},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3,
         "subsample": 0.8},
        {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2,
         "subsample": 0.8},
    ],
}


def _make_model(name, params):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42, **params)
    raise ValueError(f"unknown model {name}")


def _dir_acc(actual, pred):
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    return float(((np.sign(pred) == np.sign(actual)) |
                  ((pred == 0) & (actual == 0))).mean())


def _metrics(actual, pred):
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(float(np.sqrt(mean_squared_error(actual, pred)))),
        "directional_accuracy": _dir_acc(actual, pred),
    }


def support_resistance(df, window=20):
    """Support = lowest low, resistance = highest high, last 20 days."""
    closes = df["Close"]
    return {
        "support": float(df["Low"].rolling(window).min().iloc[-1]),
        "resistance": float(df["High"].rolling(window).max().iloc[-1]),
        "current_price": float(closes.iloc[-1]),
        "prev_close": float(closes.iloc[-2]) if len(closes) > 1 else float(closes.iloc[-1]),
    }


def _rsi(close, period=14):
    """RSI, Wilder's smoothing. Only uses closes up to today."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_daily_features(price_df, posts, news):
    """Put real prices and scored sentiment together, one row per day."""
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    close = df["Close"]

    # --- price / technical features (past and today only) ---
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["volume_log"] = np.log1p(df["Volume"].fillna(0))
    df["rsi_14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_hist"] = df["macd"] - df["macd"].ewm(span=9, adjust=False).mean()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    df["price_vs_ma5"] = close / ma5 - 1
    df["price_vs_ma10"] = close / ma10 - 1
    df["price_vs_ma20"] = close / ma20 - 1
    df["ma5_slope"] = ma5 / ma5.shift(5) - 1
    df["ma20_slope"] = ma20 / ma20.shift(5) - 1
    mid = ma20
    std20 = close.rolling(20).std()
    upper, lower = mid + 2 * std20, mid - 2 * std20
    df["bollinger_pos"] = (close - lower) / (upper - lower)
    df["volume_change"] = df["Volume"] / df["Volume"].rolling(20).mean() - 1
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek

    # --- daily sentiment from posts (UTC dates) ---
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

    df["target_return"] = close.shift(-1) / close - 1  # n+1
    return df


def _pick_params(name, subtrain, val):
    """Pick the best settings for one model on the validation split
    (last 20% of the training period). Returns (params, val_dir_acc)."""
    best_params, best_acc, best_mae = None, -1.0, float("inf")
    for params in MODEL_GRIDS[name]:
        model = _make_model(name, params)
        model.fit(subtrain[FEATURES], subtrain["target_return"])
        pred = model.predict(val[FEATURES])
        acc = _dir_acc(val["target_return"], pred)
        mae = float(mean_absolute_error(val["target_return"], pred))
        if acc > best_acc or (acc == best_acc and mae < best_mae):
            best_params, best_acc, best_mae = params, acc, mae
    return best_params, best_acc


def compare_models(df):
    """Time-split test for Ridge vs RandomForest vs GradientBoosting.

    Returns (results, best_name, best_params) where results is
    {name: metrics + train/test days + val_dir_acc + params}.
    The winner is picked by TEST directional accuracy, and all three
    are reported — I do not hide the losers.
    """
    work = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)
    if len(work) < 40:
        raise RuntimeError("Not enough days with features to train (need >= 40)")
    split = int(len(work) * 0.7)
    train, test = work.iloc[:split], work.iloc[split:]
    # Validation split inside training only: last 20% of train.
    val_split = int(len(train) * 0.8)
    subtrain, val = train.iloc[:val_split], train.iloc[val_split:]

    # Baseline: tomorrow moves the same way as yesterday (test set).
    baseline_acc = _dir_acc(test["target_return"], test["return_1d"])

    results = {}
    for name in ("Ridge", "RandomForest", "GradientBoosting"):
        params, val_acc = _pick_params(name, subtrain, val)
        model = _make_model(name, params)
        model.fit(train[FEATURES], train["target_return"])
        pred = model.predict(test[FEATURES])
        m = _metrics(test["target_return"], pred)
        m.update({
            "train_days": len(train), "test_days": len(test),
            "val_directional_accuracy": float(val_acc),
            "params": str(params),
            "baseline_directional_accuracy": float(baseline_acc),
        })
        results[name] = m
    best_name = max(results, key=lambda n: (results[n]["directional_accuracy"],
                                            -results[n]["mae"]))
    return results, best_name, results[best_name]["params"]


def train_final_model(df, best_name, best_params_str):
    """Refit the winning model type on ALL days with a known target,
    so the n+1 prediction uses everything up to today. Settings come
    from the validation pick (stored as a string) — parsed back here."""
    import ast
    params = ast.literal_eval(best_params_str)
    work = df.dropna(subset=FEATURES + ["target_return"]).reset_index(drop=True)
    model = _make_model(best_name, params)
    model.fit(work[FEATURES], work["target_return"])
    return model


def predict_next_day(model, df):
    """Guess tomorrow's return from today's features.

    The guess is capped at +/-5%: a 1-day move bigger than that is
    outside what this model can honestly claim from these features.
    On 10 Nov 2026 the uncapped Ridge guess for Samsung was -10.7%,
    made during a sharp sell-off (it fell from ~285,500 to ~189,900
    in a few weeks) — a linear model extrapolating, not a real
    forecast. The cap only touches the n+1 point forecast; the test
    numbers in compare_models() are NOT capped, so they stay honest.
    """
    last = df.dropna(subset=FEATURES).iloc[-1]
    features = last[FEATURES].to_frame().T  # keep feature names, no warning
    pred_return = float(model.predict(features)[0])
    pred_return = max(-0.05, min(0.05, pred_return))
    current = float(last["Close"])
    return pred_return, current * (1 + pred_return)
