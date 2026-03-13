import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import yfinance as yf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import lightgbm as lgb


def download_market_data(ticker="SPY", period="60d", interval="5m"):
    raw = yf.download(ticker, period=period, interval=interval, progress=False)
    raw.columns = [col[0].lower() for col in raw.columns]
    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index.name = "datetime"
    df = df.reset_index()
    df = df.dropna(subset=["close"])
    df = df[df["volume"] > 0]
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Bars loaded: {len(df):,}  |  {df['datetime'].iloc[0]}  ->  {df['datetime'].iloc[-1]}")
    return df


def compute_metrics(df, vol_window=20):
    out = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    out["log_return"]   = np.log(df["close"] / df["close"].shift(1))
    out["realised_vol"] = out["log_return"].rolling(vol_window).std()
    out["price_range"]  = (df["high"] - df["low"]) / df["close"]
    out["volume_ratio"] = df["volume"] / df["volume"].rolling(vol_window).mean().replace(0, np.nan)
    out["momentum"]     = df["close"] / df["close"].rolling(vol_window).mean() - 1.0
    metric_cols = ["log_return", "realised_vol", "price_range", "volume_ratio", "momentum"]
    out[metric_cols] = out[metric_cols].ffill().bfill()
    return out


def label_spikes(df, H=10, spike_mult=2.0, baseline_window=60):
    out = df.copy()
    out["baseline_vol"] = out["log_return"].rolling(baseline_window, min_periods=2).std().shift(1)
    reversed_ret = out["log_return"].iloc[::-1]
    out["forward_vol"] = reversed_ret.rolling(H, min_periods=2).std().iloc[::-1].values
    out["is_spike"] = (out["forward_vol"] > spike_mult * out["baseline_vol"]).astype(np.int8)
    out = out.iloc[baseline_window:-H].dropna(subset=["baseline_vol", "forward_vol"]).reset_index(drop=True)
    print(f"Bars after labelling: {len(out):,}  |  Spikes: {out['is_spike'].sum():,} ({100*out['is_spike'].mean():.1f}%)")
    return out


METRIC_COLS = ["log_return", "realised_vol", "price_range", "volume_ratio", "momentum"]
SOFT_THRESHOLDS = [0.002, 0.003, 0.004, 1.5, 0.005]


def linear_trend(x):
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    t_mu, x_mu = (n - 1) / 2.0, x.mean()
    num = ((t - t_mu) * (x - x_mu)).sum()
    den = ((t - t_mu) ** 2).sum()
    return num / den if den > 0 else 0.0


def extract_features(window):
    feats = []
    for i in range(window.shape[1]):
        x = window[:, i]
        feats.extend([x.mean(), x.std(), x.min(), x.max(), x[-1],
                      linear_trend(x), (x > SOFT_THRESHOLDS[i]).mean()])
    return np.array(feats, dtype=np.float32)


def make_feature_names():
    stats = ["mean", "std", "min", "max", "last", "slope", "frac_above"]
    return [f"{m}__{s}" for m in METRIC_COLS for s in stats]


def build_windows(df, W):
    values = df[METRIC_COLS].values
    labels = df["is_spike"].values
    n_samples = len(df) - W
    X = np.zeros((n_samples, len(METRIC_COLS) * 7), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int8)
    for i in range(n_samples):
        X[i] = extract_features(values[i:i + W])
        y[i] = labels[i + W]
    return X, y


def temporal_split(X, y, val_frac=0.10, test_frac=0.20):
    n = len(X)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_test - n_val
    X_train, y_train = X[:n_train],              y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],        y[n_train+n_val:]
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:<6}: {len(ys):>5,} samples | spikes: {ys.sum():>4,} ({100*ys.mean():.1f}%)")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, X_val, y_val):
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)
    print(f"  scale_pos_weight = {spw:.1f}")
    params = {
        "objective": "binary", "metric": "binary_logloss",
        "scale_pos_weight": spw, "num_leaves": 31,
        "min_child_samples": 20, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "verbose": -1, "seed": 42,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=1_000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )
    return model


def choose_threshold(model, X_val, y_val):
    probs = model.predict(X_val)
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.argmax(f1[:-1]))
    print(f"  Best threshold = {thresholds[best]:.4f}  ->  val F1 = {f1[best]:.4f}")
    return float(thresholds[best])


def evaluate(model, X_test, y_test, thresh, feature_names):
    probs = model.predict(X_test)
    preds = (probs >= thresh).astype(int)
    roc  = roc_auc_score(y_test, probs)
    pra  = average_precision_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    cm   = confusion_matrix(y_test, preds)
    print(f"\n{'='*50}\nTEST SET RESULTS\n{'='*50}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  PR-AUC    : {pra:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Threshold : {thresh:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"    FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n  Top-10 features by gain:")
    for _, row in imp.head(10).iterrows():
        bar = "#" * int(30 * row.importance / imp.importance.max())
        print(f"    {row.feature:<35s}  {bar}")
    return {"roc_auc": roc, "pr_auc": pra, "precision": prec,
            "recall": rec, "f1": f1, "probs": probs, "preds": preds,
            "cm": cm.tolist(), "feature_importance": imp}


if __name__ == "__main__":
    np.random.seed(42)

    TICKER, PERIOD, INTERVAL = "SPY", "60d", "5m"
    W, H, SPIKE_MULT, VOL_WINDOW = 30, 10, 2.0, 20

    df_raw      = download_market_data(TICKER, PERIOD, INTERVAL)
    df_metrics  = compute_metrics(df_raw, vol_window=VOL_WINDOW)
    df_labelled = label_spikes(df_metrics, H=H, spike_mult=SPIKE_MULT, baseline_window=VOL_WINDOW)

    X, y       = build_windows(df_labelled, W=W)
    feat_names = make_feature_names()

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y)

    model   = train_model(X_train, y_train, X_val, y_val)
    thresh  = choose_threshold(model, X_val, y_val)
    results = evaluate(model, X_test, y_test, thresh, feat_names)
