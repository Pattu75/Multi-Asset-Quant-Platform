import os
from datetime import date

import mysql.connector
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mysql.connector import Error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Zakariya Boutayeb | Multi-Asset Quantitative Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# AUTHOR / BRANDING
# ============================================================
AUTHOR_NAME = "Zakariya Boutayeb"
AUTHOR_TAG = "Quantitative Finance & Data Science"
APP_TITLE = "📈 Multi-Asset Quantitative Trading & Portfolio Analytics Platform"
APP_SUBTITLE = f"Built by {AUTHOR_NAME} — {AUTHOR_TAG}"

BENCHMARKS = {
    "S&P 500 ETF (SPY)": "SPY",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Nasdaq 100 ETF (QQQ)": "QQQ",
    "Russell 2000 ETF (IWM)": "IWM",
    "US Aggregate Bond ETF (AGG)": "AGG",
}

# ============================================================
# MYSQL CONFIG
# Prefer Streamlit secrets if available, else fallback to env, else hardcoded.
# ============================================================
def get_db_config():
    try:
        if "mysql" in st.secrets:
            return {
                "host": st.secrets["mysql"]["host"],
                "port": int(st.secrets["mysql"]["port"]),
                "user": st.secrets["mysql"]["user"],
                "password": st.secrets["mysql"]["password"],
                "database": st.secrets["mysql"]["database"],
            }
    except Exception:
        pass

    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", 3306)),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "**********"),
        "database": os.getenv("MYSQL_DATABASE", "Quant_Platform"),
    }


DB_CONFIG = get_db_config()

# ============================================================
# DB HELPERS
# ============================================================
def get_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        raise RuntimeError(f"Failed to connect to MySQL: {e}") from e


def run_query(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql(query, conn, params=params)
    finally:
        conn.close()


# ============================================================
# MYSQL LOADERS
# ============================================================
@st.cache_data(show_spinner=False)
def load_ticker_universe() -> pd.DataFrame:
    query = """
        SELECT
            ticker,
            asset_name AS name,
            asset_class AS category,
            COALESCE(sector, '') AS sector,
            COALESCE(industry, '') AS industry,
            COALESCE(exchange_name, '') AS exchange_name,
            COALESCE(currency, '') AS currency
        FROM tickers
        WHERE is_active = TRUE
        ORDER BY asset_class, asset_name
    """
    df = run_query(query)

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "name", "category", "sector", "industry", "exchange_name", "currency"]
        )

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_price_history_from_db(tickers: tuple, start_date, end_date) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"""
        SELECT
            t.ticker,
            p.trade_date,
            COALESCE(p.adj_close_price, p.close_price) AS close_price
        FROM price_history p
        JOIN tickers t
            ON p.ticker_id = t.ticker_id
        WHERE t.ticker IN ({placeholders})
          AND p.trade_date BETWEEN %s AND %s
        ORDER BY p.trade_date, t.ticker
    """
    params = list(tickers) + [start_date, end_date]
    df = run_query(query, params=params)

    if df.empty:
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    prices = df.pivot(index="trade_date", columns="ticker", values="close_price").sort_index()
    return prices.dropna(how="all")


@st.cache_data(show_spinner=False)
def load_asset_snapshot_from_db(ticker: str) -> dict:
    query = """
        SELECT
            ticker,
            asset_name,
            asset_class,
            exchange_name,
            sector,
            industry,
            currency
        FROM tickers
        WHERE ticker = %s
        LIMIT 1
    """
    df = run_query(query, params=[ticker])

    if df.empty:
        return {}

    return df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_latest_signal_table_from_db(tickers: tuple, signal_type: str = "MA_MOMENTUM") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(tickers))
    query = f"""
        SELECT
            t.ticker,
            s.signal_date,
            s.signal_type,
            s.signal_value,
            s.signal_label,
            s.short_ma,
            s.long_ma,
            s.momentum_window
        FROM signals s
        JOIN tickers t
            ON s.ticker_id = t.ticker_id
        JOIN (
            SELECT
                ticker_id,
                MAX(signal_date) AS max_signal_date
            FROM signals
            WHERE signal_type = %s
            GROUP BY ticker_id
        ) latest
            ON s.ticker_id = latest.ticker_id
           AND s.signal_date = latest.max_signal_date
        WHERE t.ticker IN ({placeholders})
          AND s.signal_type = %s
        ORDER BY
            CASE s.signal_label
                WHEN 'BUY' THEN 1
                WHEN 'HOLD' THEN 2
                WHEN 'SELL' THEN 3
                ELSE 4
            END,
            s.signal_value DESC
    """
    params = [signal_type] + list(tickers) + [signal_type]
    df = run_query(query, params=params)

    if df.empty:
        return pd.DataFrame()

    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


# ============================================================
# ADVANCED SQL QUERIES
# ============================================================
@st.cache_data(show_spinner=False)
def sql_top_momentum(limit_n: int = 25, lookback_days: int = 20) -> pd.DataFrame:
    query = """
        WITH priced AS (
            SELECT
                t.ticker,
                p.trade_date,
                COALESCE(p.adj_close_price, p.close_price) AS px,
                LAG(COALESCE(p.adj_close_price, p.close_price), %s)
                    OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS px_lag
            FROM price_history p
            JOIN tickers t ON p.ticker_id = t.ticker_id
            WHERE t.is_active = TRUE
        ),
        ranked AS (
            SELECT
                ticker,
                trade_date,
                px,
                px_lag,
                CASE
                    WHEN px_lag IS NULL OR px_lag = 0 THEN NULL
                    ELSE (px / px_lag) - 1
                END AS momentum_return,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
            FROM priced
        )
        SELECT
            ticker,
            trade_date,
            px AS latest_price,
            px_lag AS lookback_price,
            momentum_return
        FROM ranked
        WHERE rn = 1
          AND momentum_return IS NOT NULL
        ORDER BY momentum_return DESC
        LIMIT %s
    """
    return run_query(query, params=[lookback_days, limit_n])


@st.cache_data(show_spinner=False)
def sql_bottom_momentum(limit_n: int = 25, lookback_days: int = 20) -> pd.DataFrame:
    query = """
        WITH priced AS (
            SELECT
                t.ticker,
                p.trade_date,
                COALESCE(p.adj_close_price, p.close_price) AS px,
                LAG(COALESCE(p.adj_close_price, p.close_price), %s)
                    OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS px_lag
            FROM price_history p
            JOIN tickers t ON p.ticker_id = t.ticker_id
            WHERE t.is_active = TRUE
        ),
        ranked AS (
            SELECT
                ticker,
                trade_date,
                px,
                px_lag,
                CASE
                    WHEN px_lag IS NULL OR px_lag = 0 THEN NULL
                    ELSE (px / px_lag) - 1
                END AS momentum_return,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
            FROM priced
        )
        SELECT
            ticker,
            trade_date,
            px AS latest_price,
            px_lag AS lookback_price,
            momentum_return
        FROM ranked
        WHERE rn = 1
          AND momentum_return IS NOT NULL
        ORDER BY momentum_return ASC
        LIMIT %s
    """
    return run_query(query, params=[lookback_days, limit_n])


@st.cache_data(show_spinner=False)
def sql_highest_volatility(limit_n: int = 25, lookback_days: int = 20) -> pd.DataFrame:
    query = """
        WITH returns_cte AS (
            SELECT
                t.ticker,
                p.trade_date,
                COALESCE(p.adj_close_price, p.close_price) AS px,
                LAG(COALESCE(p.adj_close_price, p.close_price))
                    OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS prev_px
            FROM price_history p
            JOIN tickers t ON p.ticker_id = t.ticker_id
            WHERE t.is_active = TRUE
        ),
        daily_returns AS (
            SELECT
                ticker,
                trade_date,
                CASE
                    WHEN prev_px IS NULL OR prev_px = 0 THEN NULL
                    ELSE (px / prev_px) - 1
                END AS daily_ret
            FROM returns_cte
        ),
        latest_date AS (
            SELECT MAX(trade_date) AS max_dt FROM daily_returns
        )
        SELECT
            dr.ticker,
            COUNT(*) AS observations,
            STDDEV_SAMP(dr.daily_ret) * SQRT(252) AS annualized_volatility
        FROM daily_returns dr
        CROSS JOIN latest_date ld
        WHERE dr.daily_ret IS NOT NULL
          AND dr.trade_date >= DATE_SUB(ld.max_dt, INTERVAL %s DAY)
        GROUP BY dr.ticker
        HAVING COUNT(*) >= 10
        ORDER BY annualized_volatility DESC
        LIMIT %s
    """
    return run_query(query, params=[lookback_days, limit_n])


@st.cache_data(show_spinner=False)
def sql_latest_prices(limit_n: int = 50) -> pd.DataFrame:
    query = """
        WITH ranked AS (
            SELECT
                t.ticker,
                t.asset_name,
                t.asset_class,
                p.trade_date,
                COALESCE(p.adj_close_price, p.close_price) AS close_price,
                ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY p.trade_date DESC) AS rn
            FROM tickers t
            JOIN price_history p ON t.ticker_id = p.ticker_id
            WHERE t.is_active = TRUE
        )
        SELECT
            ticker,
            asset_name,
            asset_class,
            trade_date,
            close_price
        FROM ranked
        WHERE rn = 1
        ORDER BY ticker
        LIMIT %s
    """
    return run_query(query, params=[limit_n])


@st.cache_data(show_spinner=False)
def sql_signal_summary() -> pd.DataFrame:
    query = """
        WITH latest_signals AS (
            SELECT
                s.ticker_id,
                s.signal_label,
                ROW_NUMBER() OVER (PARTITION BY s.ticker_id ORDER BY s.signal_date DESC) AS rn
            FROM signals s
            WHERE s.signal_type = 'MA_MOMENTUM'
        )
        SELECT
            ls.signal_label,
            COUNT(*) AS ticker_count
        FROM latest_signals ls
        WHERE ls.rn = 1
        GROUP BY ls.signal_label
        ORDER BY ticker_count DESC
    """
    return run_query(query)


# ============================================================
# FORMATTERS
# ============================================================
def format_large_number(x):
    if x is None or pd.isna(x):
        return "N/A"
    try:
        x = float(x)
    except Exception:
        return "N/A"

    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"${x / 1e12:.2f}T"
    if abs_x >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs_x >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x:,.0f}"


def format_percent(x, decimals=2):
    if x is None or pd.isna(x):
        return "N/A"
    try:
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "N/A"


def format_number(x, decimals=2, prefix=""):
    if x is None or pd.isna(x):
        return "N/A"
    try:
        return f"{prefix}{float(x):,.{decimals}f}"
    except Exception:
        return "N/A"


def search_labels(df: pd.DataFrame, search_text: str) -> list[str]:
    df = df.copy()
    df["label"] = df["name"] + " (" + df["ticker"] + ")"

    if not search_text.strip():
        return df["label"].tolist()

    term = search_text.strip().lower()
    mask = (
        df["name"].str.lower().str.contains(term, na=False)
        | df["ticker"].str.lower().str.contains(term, na=False)
        | df["category"].str.lower().str.contains(term, na=False)
    )
    return df.loc[mask, "label"].tolist()

# ============================================================
# ANALYTICS HELPERS
# ============================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def annualized_return(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    total_return = (1 + series).prod()
    return total_return ** (252 / len(series)) - 1


def annualized_volatility(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return series.std() * np.sqrt(252)


def sharpe_ratio(series: pd.Series, risk_free_rate: float = 0.0) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    excess = series - risk_free_rate / 252
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return excess.mean() / std * np.sqrt(252)


def max_drawdown(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def make_growth_curve(series: pd.Series, initial_capital: float = 10000.0) -> pd.Series:
    series = series.dropna()
    return initial_capital * (1 + series).cumprod()


def performance_table(returns_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns_df.columns:
        s = returns_df[col].dropna()
        rows.append(
            {
                "Asset": col,
                "Annual Return": annualized_return(s),
                "Annual Volatility": annualized_volatility(s),
                "Sharpe Ratio": sharpe_ratio(s),
                "Max Drawdown": max_drawdown(s),
            }
        )
    return pd.DataFrame(rows)


def compute_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan

    df.columns = ["asset", "benchmark"]
    bench_var = df["benchmark"].var()
    if pd.isna(bench_var) or bench_var == 0:
        return np.nan

    cov = df["asset"].cov(df["benchmark"])
    return cov / bench_var


def compute_capm_expected_return(
    beta: float, benchmark_returns: pd.Series, risk_free_rate: float = 0.02
) -> float:
    if pd.isna(beta):
        return np.nan

    benchmark_ann_return = annualized_return(benchmark_returns.dropna())
    if pd.isna(benchmark_ann_return):
        return np.nan

    return risk_free_rate + beta * (benchmark_ann_return - risk_free_rate)


def compute_alpha(
    asset_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.02
) -> float:
    beta = compute_beta(asset_returns, benchmark_returns)
    capm_return = compute_capm_expected_return(beta, benchmark_returns, risk_free_rate=risk_free_rate)
    realized_return = annualized_return(asset_returns.dropna())

    if pd.isna(capm_return) or pd.isna(realized_return):
        return np.nan

    return realized_return - capm_return


def compute_tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return np.nan

    df.columns = ["asset", "benchmark"]
    active_returns = df["asset"] - df["benchmark"]
    return active_returns.std() * np.sqrt(252)


def compute_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return np.nan

    df.columns = ["asset", "benchmark"]
    active_returns = df["asset"] - df["benchmark"]

    te = active_returns.std() * np.sqrt(252)
    if pd.isna(te) or te == 0:
        return np.nan

    active_ann_return = annualized_return(active_returns)
    return active_ann_return / te


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    df.columns = ["asset", "benchmark"]
    cov = df["asset"].rolling(window).cov(df["benchmark"])
    var = df["benchmark"].rolling(window).var()
    beta_series = cov / var
    beta_series.name = "Rolling Beta"
    return beta_series


def compute_factor_exposures(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    lookback_momentum: int = 60
) -> dict:
    beta = compute_beta(asset_returns, benchmark_returns)

    momentum_proxy = np.nan
    if len(asset_returns.dropna()) >= lookback_momentum:
        momentum_proxy = (1 + asset_returns.dropna().tail(lookback_momentum)).prod() - 1

    vol = annualized_volatility(asset_returns.dropna())

    return {
        "market_beta": beta,
        "momentum_proxy": momentum_proxy,
        "volatility": vol,
    }


# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(price_series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": price_series.copy()})
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    return df


# ============================================================
# OPTIMIZER / PORTFOLIO HELPERS
# ============================================================
def optimize_max_sharpe(returns_df: pd.DataFrame, n_portfolios: int = 5000, risk_free_rate: float = 0.0):
    returns_df = returns_df.dropna()
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    results = []
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()

        portfolio_return = float(np.sum(mean_returns * weights))
        portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))

        if portfolio_vol == 0:
            sharpe = np.nan
        else:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol

        results.append(
            {
                "Return": portfolio_return,
                "Volatility": portfolio_vol,
                "Sharpe": sharpe,
                "Weights": weights,
            }
        )

    results_df = pd.DataFrame(results).dropna(subset=["Sharpe"])
    if results_df.empty:
        return pd.DataFrame(), None, pd.DataFrame()

    best_idx = results_df["Sharpe"].idxmax()
    best_portfolio = results_df.loc[best_idx]

    best_weights = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": best_portfolio["Weights"],
        }
    ).sort_values("Weight", ascending=False)

    return results_df, best_portfolio, best_weights


def portfolio_returns_from_weights(returns_df: pd.DataFrame, weights_df: pd.DataFrame) -> pd.Series:
    weights_map = dict(zip(weights_df["Ticker"], weights_df["Weight"]))
    weights = np.array([weights_map.get(col, 0.0) for col in returns_df.columns])
    return pd.Series(np.dot(returns_df.values, weights), index=returns_df.index, name="Optimized Portfolio")


def normalize_weight_inputs(weight_inputs: dict) -> dict:
    clean_weights = {k: float(v) for k, v in weight_inputs.items() if float(v) > 0}
    total = sum(clean_weights.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean_weights.items()}


def get_rebalance_groups(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    if frequency == "Weekly":
        return pd.Series(index.to_period("W-FRI"), index=index)
    if frequency == "Monthly":
        return pd.Series(index.to_period("M"), index=index)
    return pd.Series(index, index=index)


def backtest_custom_portfolio(
    prices: pd.DataFrame,
    weights_map: dict,
    rebalance_frequency: str = "Monthly",
):
    prices = prices.dropna(how="any").copy()
    if prices.empty:
        return {
            "values": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "weights": pd.Series(dtype=float),
        }

    weights = pd.Series(weights_map, dtype=float)
    weights = weights.reindex(prices.columns).fillna(0.0)

    if weights.sum() <= 0:
        return {
            "values": pd.Series(dtype=float),
            "returns": pd.Series(dtype=float),
            "drawdown": pd.Series(dtype=float),
            "weights": pd.Series(dtype=float),
        }

    weights = weights / weights.sum()

    if rebalance_frequency == "Daily":
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_values = (1 + portfolio_returns).cumprod().rename("Custom Portfolio")
    else:
        portfolio_values_list = []
        current_value = 1.0
        shares = None
        prev_group = None
        groups = get_rebalance_groups(prices.index, rebalance_frequency)

        for dt, row in prices.iterrows():
            current_group = groups.loc[dt]

            if shares is None or current_group != prev_group:
                shares = (current_value * weights) / row
                prev_group = current_group

            current_value = float((shares * row).sum())
            portfolio_values_list.append(current_value)

        portfolio_values = pd.Series(
            portfolio_values_list,
            index=prices.index,
            name="Custom Portfolio"
        )
        portfolio_returns = portfolio_values.pct_change().dropna()

    rolling_max = portfolio_values.cummax()
    drawdown = portfolio_values / rolling_max - 1

    return {
        "values": portfolio_values,
        "returns": portfolio_returns,
        "drawdown": drawdown,
        "weights": weights.sort_values(ascending=False),
    }


# ============================================================
# SIGNALS / STRATEGY HELPERS
# ============================================================
def generate_signals(prices: pd.DataFrame, short_ma: int = 50, long_ma: int = 200, momentum_window: int = 60):
    short_avg = prices.rolling(short_ma).mean()
    long_avg = prices.rolling(long_ma).mean()
    momentum = prices.pct_change(momentum_window)

    signals = pd.DataFrame(index=prices.index, columns=prices.columns)

    for col in prices.columns:
        buy_condition = (short_avg[col] > long_avg[col]) & (momentum[col] > 0)
        sell_condition = (short_avg[col] < long_avg[col]) & (momentum[col] < 0)

        signals[col] = np.where(
            buy_condition,
            "BUY",
            np.where(sell_condition, "SELL", "HOLD")
        )

    return signals


def latest_signal_table(prices: pd.DataFrame, short_ma: int = 50, long_ma: int = 200, momentum_window: int = 60):
    signals = generate_signals(prices, short_ma=short_ma, long_ma=long_ma, momentum_window=momentum_window)
    returns = prices.pct_change()

    valid_signals = signals.dropna(how="all")
    if valid_signals.empty:
        return pd.DataFrame()

    latest_date = valid_signals.index[-1]

    latest = pd.DataFrame(
        {
            "Ticker": signals.columns,
            "Signal": signals.loc[latest_date].values,
            "Last Price": prices.loc[latest_date].values,
            "60D Return": prices.pct_change(momentum_window).loc[latest_date].values,
            "20D Vol": returns.rolling(20).std().loc[latest_date].values,
        }
    )

    signal_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    latest["SignalRank"] = latest["Signal"].map(signal_order)
    latest = latest.sort_values(["SignalRank", "60D Return"], ascending=[True, False]).drop(columns=["SignalRank"])

    return latest


def build_momentum_strategy(
    prices: pd.DataFrame,
    lookback: int = 60,
    vol_window: int = 20,
    top_n: int = 3,
    target_vol: float = 0.10,
    use_regime_filter: bool = True,
):
    prices = prices.dropna(how="all")
    returns = compute_returns(prices)
    momentum = prices.pct_change(lookback)
    rolling_vol = returns.rolling(vol_window).std()

    raw_weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    for dt in returns.index:
        mom_today = momentum.loc[dt].dropna() if dt in momentum.index else pd.Series(dtype=float)
        vol_today = rolling_vol.loc[dt].dropna() if dt in rolling_vol.index else pd.Series(dtype=float)

        valid = mom_today.index.intersection(vol_today.index)
        if len(valid) == 0:
            continue

        mom_today = mom_today.loc[valid]
        vol_today = vol_today.loc[valid]

        selected = mom_today.sort_values(ascending=False).head(top_n)
        selected = selected[selected > 0]

        if len(selected) == 0:
            continue

        inv_vol = 1 / vol_today.loc[selected.index]
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
        if len(inv_vol) == 0 or inv_vol.sum() == 0:
            continue

        weights = inv_vol / inv_vol.sum()
        raw_weights.loc[dt, weights.index] = weights.values

    if use_regime_filter and "SPY" in prices.columns:
        spy_ma200 = prices["SPY"].rolling(200).mean()
        regime = (prices["SPY"] > spy_ma200).reindex(raw_weights.index).fillna(False)
        raw_weights = raw_weights.mul(regime.astype(float), axis=0)

    shifted_weights = raw_weights.shift(1).fillna(0.0)
    strategy_returns = (shifted_weights * returns).sum(axis=1)

    realized_vol = strategy_returns.rolling(20).std() * np.sqrt(252)
    vol_scale = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vol_scale = vol_scale.clip(lower=0.0, upper=2.0)
    scaled_returns = strategy_returns * vol_scale.shift(1).fillna(1.0)

    cumulative = (1 + scaled_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    rolling_sharpe = (scaled_returns.rolling(60).mean() / scaled_returns.rolling(60).std()) * np.sqrt(252)

    return {
        "returns": scaled_returns,
        "weights": raw_weights,
        "momentum": momentum,
        "rolling_vol": rolling_vol,
        "drawdown": drawdown,
        "rolling_sharpe": rolling_sharpe,
    }


# ============================================================
# ALPHA RESEARCH LAB HELPERS
# ============================================================
def compute_factor_signal(prices: pd.DataFrame, signal_name: str, lookback: int = 60, vol_window: int = 20) -> pd.DataFrame:
    returns = compute_returns(prices)

    if signal_name == "Momentum":
        signal = prices.pct_change(lookback)
    elif signal_name == "Volatility Adjusted Momentum":
        raw_mom = prices.pct_change(lookback)
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        signal = raw_mom / rolling_vol
    elif signal_name == "Mean Reversion":
        signal = -prices.pct_change(lookback)
    elif signal_name == "Short-Term Reversal":
        signal = -prices.pct_change(5)
    else:
        signal = prices.pct_change(lookback)

    return signal.replace([np.inf, -np.inf], np.nan)


def compute_forward_returns(prices: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    return prices.shift(-horizon) / prices - 1


def cross_sectional_rank(signal_df: pd.DataFrame) -> pd.DataFrame:
    return signal_df.rank(axis=1, pct=True)


def compute_daily_ic(signal_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> pd.Series:
    common_index = signal_df.index.intersection(forward_returns_df.index)
    ic_values = []

    for dt in common_index:
        s = signal_df.loc[dt]
        f = forward_returns_df.loc[dt]
        pair = pd.concat([s, f], axis=1).dropna()

        if len(pair) < 3:
            ic_values.append(np.nan)
        else:
            ranked_signal = pair.iloc[:, 0].rank()
            ranked_forward = pair.iloc[:, 1].rank()
            ic = ranked_signal.corr(ranked_forward)
            ic_values.append(ic)

    return pd.Series(ic_values, index=common_index, name="Daily IC")


def compute_hit_rate(signal_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> float:
    merged = []
    common_index = signal_df.index.intersection(forward_returns_df.index)

    for dt in common_index:
        s = signal_df.loc[dt]
        f = forward_returns_df.loc[dt]
        pair = pd.concat([s, f], axis=1).dropna()
        if len(pair) == 0:
            continue

        pair.columns = ["signal", "forward_return"]
        hit = np.sign(pair["signal"]) == np.sign(pair["forward_return"])
        merged.extend(hit.astype(float).tolist())

    if len(merged) == 0:
        return np.nan
    return float(np.mean(merged))


def compute_quantile_spread(signal_df: pd.DataFrame, forward_returns_df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    rows = []
    common_index = signal_df.index.intersection(forward_returns_df.index)

    for dt in common_index:
        s = signal_df.loc[dt]
        f = forward_returns_df.loc[dt]
        pair = pd.concat([s, f], axis=1).dropna()
        if len(pair) < n_buckets:
            continue

        pair.columns = ["signal", "forward_return"]
        try:
            pair["bucket"] = pd.qcut(pair["signal"], q=n_buckets, labels=False, duplicates="drop")
        except Exception:
            continue

        grouped = pair.groupby("bucket")["forward_return"].mean()
        row = {"Date": dt}
        for b, val in grouped.items():
            row[f"Bucket_{int(b)+1}"] = val

        if f"Bucket_{n_buckets}" in row and "Bucket_1" in row:
            row["Top-Bottom Spread"] = row[f"Bucket_{n_buckets}"] - row["Bucket_1"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    spread_df = pd.DataFrame(rows).set_index("Date").sort_index()
    return spread_df

# ============================================================
# LOAD UNIVERSE
# ============================================================
universe_df = load_ticker_universe()

if universe_df.empty:
    st.error("No tickers found in MySQL. Please run your ingestion script first.")
    st.stop()

all_categories = sorted(universe_df["category"].dropna().unique().tolist())
search_choices = [f"{row['ticker']} | {row['name']} | {row['category']}" for _, row in universe_df.iterrows()]

# ============================================================
# HEADER
# ============================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a page",
    [
        "Research Terminal",
        "Market Data Explorer",
        "Multi-Asset Comparison",
        "Portfolio Optimizer",
        "Custom Portfolio Builder",
        "Alpha Research Lab",
        "Signal Dashboard",
        "Quant Strategy Backtester",
        "Advanced SQL Lab",
    ],
)

st.sidebar.header("Global Settings")
start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
benchmark_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
benchmark_ticker = BENCHMARKS[benchmark_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")
st.sidebar.markdown(f"Using MySQL database: `{DB_CONFIG['database']}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Universe")
st.sidebar.metric("Tickers loaded", f"{len(universe_df):,}")
st.sidebar.metric("Categories", f"{len(all_categories):,}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Author")
st.sidebar.markdown(AUTHOR_NAME)
st.sidebar.markdown(AUTHOR_TAG)
st.sidebar.markdown("GitHub: Pattu75")

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

# ============================================================
# PAGE 0: RESEARCH TERMINAL
# ============================================================
if page == "Research Terminal":
    st.subheader("Market Research Terminal")
    st.caption("MySQL-backed multi-asset research interface")

    search_term = st.text_input(
        "Search by ticker, company name, or category",
        placeholder="Examples: AAPL, NVIDIA, ETF, Bond, REIT"
    )

    filtered_universe = universe_df.copy()
    if search_term.strip():
        term = search_term.strip().lower()
        mask = (
            filtered_universe["ticker"].str.lower().str.contains(term, na=False)
            | filtered_universe["name"].str.lower().str.contains(term, na=False)
            | filtered_universe["category"].str.lower().str.contains(term, na=False)
        )
        filtered_universe = filtered_universe.loc[mask].copy()

    col_a, col_b = st.columns([1.1, 1.9])

    with col_a:
        st.markdown("### Search Results")
        st.dataframe(filtered_universe, use_container_width=True, height=500)
        st.download_button(
            label="Download ticker universe as CSV",
            data=filtered_universe.to_csv(index=False).encode("utf-8"),
            file_name="ticker_universe.csv",
            mime="text/csv",
        )

    with col_b:
        selection = st.selectbox("Open asset", options=search_choices, index=0)
        selected_ticker = selection.split(" | ")[0]

        with st.spinner("Loading asset data from MySQL..."):
            terminal_prices = load_price_history_from_db((selected_ticker, benchmark_ticker), start_date, end_date)
            snapshot = load_asset_snapshot_from_db(selected_ticker)

        if terminal_prices.empty or selected_ticker not in terminal_prices.columns:
            st.warning("No data returned from MySQL for this asset.")
        else:
            asset_prices = terminal_prices[[selected_ticker]].dropna()
            asset_returns = compute_returns(asset_prices)[selected_ticker]

            benchmark_prices = terminal_prices[[benchmark_ticker]].dropna() if benchmark_ticker in terminal_prices.columns else pd.DataFrame()
            benchmark_returns = compute_returns(benchmark_prices).iloc[:, 0] if not benchmark_prices.empty else pd.Series(dtype=float)

            beta_value = compute_beta(asset_returns, benchmark_returns) if not benchmark_returns.empty else np.nan
            capm_return = compute_capm_expected_return(beta_value, benchmark_returns, risk_free_rate=0.02) if not benchmark_returns.empty else np.nan
            alpha_value = compute_alpha(asset_returns, benchmark_returns, risk_free_rate=0.02) if not benchmark_returns.empty else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ticker", selected_ticker)
            m2.metric("Annual Return", f"{annualized_return(asset_returns):.2%}")
            m3.metric("Annual Volatility", f"{annualized_volatility(asset_returns):.2%}")
            m4.metric("Sharpe Ratio", f"{sharpe_ratio(asset_returns):.2f}")

            st.markdown("## Asset Snapshot")
            s1, s2, s3 = st.columns(3)
            s1.metric("Asset Name", snapshot.get("asset_name", "N/A"))
            s2.metric("Category", snapshot.get("asset_class", "N/A"))
            s3.metric("Exchange", snapshot.get("exchange_name", "N/A"))

            s4, s5, s6 = st.columns(3)
            s4.metric("Sector", snapshot.get("sector", "N/A"))
            s5.metric("Industry", snapshot.get("industry", "N/A"))
            s6.metric("Currency", snapshot.get("currency", "N/A"))

            c1, c2 = st.columns(2)
            c1.metric("Beta (Hist.)", format_number(beta_value))
            c2.metric("Expected Return (CAPM)", format_percent(capm_return))

            c3, c4 = st.columns(2)
            c3.metric("Alpha vs CAPM", format_percent(alpha_value))
            c4.metric("Max Drawdown", format_percent(max_drawdown(asset_returns)))

            fig_term = px.line(asset_prices, x=asset_prices.index, y=selected_ticker, title=f"{selected_ticker} Price History")
            fig_term.update_layout(height=420, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_term, use_container_width=True)

            if benchmark_ticker in terminal_prices.columns:
                combined = terminal_prices[[selected_ticker, benchmark_ticker]].dropna()
                if not combined.empty:
                    normalized = combined / combined.iloc[0] * 100
                    fig_bench = px.line(
                        normalized,
                        x=normalized.index,
                        y=normalized.columns,
                        title="Normalized Performance vs Benchmark"
                    )
                    fig_bench.update_layout(height=420, xaxis_title="Date", yaxis_title="Base = 100")
                    st.plotly_chart(fig_bench, use_container_width=True)

            st.download_button(
                label="Download selected asset prices",
                data=asset_prices.to_csv().encode("utf-8"),
                file_name=f"{selected_ticker}_prices.csv",
                mime="text/csv",
            )

# ============================================================
# PAGE 1: MARKET DATA EXPLORER
# ============================================================
elif page == "Market Data Explorer":
    st.subheader("Market Data Explorer")

    asset_class = st.selectbox("Select asset class", all_categories)

    class_df = universe_df[universe_df["category"] == asset_class].copy()
    search_asset = st.text_input(
        "Search within selected asset class",
        placeholder="Type ticker or company name"
    )
    class_options = search_labels(class_df, search_asset)

    if not class_options:
        st.warning("No assets match your search.")
    else:
        selected_label = st.selectbox("Select asset", class_options)
        ticker = selected_label.split("(")[-1].replace(")", "").strip()

        with st.spinner("Loading market data from MySQL..."):
            prices = load_price_history_from_db((ticker, benchmark_ticker), start_date, end_date)
            snapshot = load_asset_snapshot_from_db(ticker)

        if prices.empty or ticker not in prices.columns:
            st.warning("No data returned for the selected ticker and date range.")
        else:
            asset_prices = prices[[ticker]].dropna()
            asset_returns = compute_returns(asset_prices)[ticker]
            tech_df = add_technical_indicators(asset_prices[ticker])

            benchmark_prices = prices[[benchmark_ticker]].dropna() if benchmark_ticker in prices.columns else pd.DataFrame()
            benchmark_returns = compute_returns(benchmark_prices).iloc[:, 0] if not benchmark_prices.empty else pd.Series(dtype=float)

            beta_value = compute_beta(asset_returns, benchmark_returns) if not benchmark_returns.empty else np.nan
            capm_return = compute_capm_expected_return(beta_value, benchmark_returns, risk_free_rate=0.02) if not benchmark_returns.empty else np.nan
            alpha_value = compute_alpha(asset_returns, benchmark_returns, risk_free_rate=0.02) if not benchmark_returns.empty else np.nan
            tracking_error = compute_tracking_error(asset_returns, benchmark_returns) if not benchmark_returns.empty else np.nan
            factor_exposures = compute_factor_exposures(asset_returns, benchmark_returns) if not benchmark_returns.empty else {}
            rolling_beta_series = rolling_beta(asset_returns, benchmark_returns, window=60) if not benchmark_returns.empty else pd.Series(dtype=float)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ticker", ticker)
            c2.metric("Annual Return", f"{annualized_return(asset_returns):.2%}")
            c3.metric("Annual Volatility", f"{annualized_volatility(asset_returns):.2%}")
            c4.metric("Sharpe Ratio", f"{sharpe_ratio(asset_returns):.2f}")

            st.markdown("## Asset Snapshot")
            s1, s2, s3 = st.columns(3)
            s1.metric("Asset Name", snapshot.get("asset_name", "N/A"))
            s2.metric("Category", snapshot.get("asset_class", "N/A"))
            s3.metric("Exchange", snapshot.get("exchange_name", "N/A"))

            s4, s5, s6 = st.columns(3)
            s4.metric("Sector", snapshot.get("sector", "N/A"))
            s5.metric("Industry", snapshot.get("industry", "N/A"))
            s6.metric("Currency", snapshot.get("currency", "N/A"))

            st.markdown("## CAPM Analytics")
            cap1, cap2, cap3, cap4 = st.columns(4)
            cap1.metric("Beta (Hist.)", format_number(beta_value))
            cap2.metric("Expected Return (CAPM)", format_percent(capm_return))
            cap3.metric("Alpha vs CAPM", format_percent(alpha_value))
            cap4.metric("Tracking Error", format_percent(tracking_error))

            st.markdown("## Factor Analysis")
            f1, f2, f3 = st.columns(3)
            f1.metric("Market Beta", format_number(factor_exposures.get("market_beta")))
            f2.metric("Momentum Proxy", format_percent(factor_exposures.get("momentum_proxy")))
            f3.metric("Volatility", format_percent(factor_exposures.get("volatility")))

            st.markdown("## 📊 Price Chart with Moving Averages")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=tech_df.index, y=tech_df["Close"], mode="lines", name=ticker))
            fig_price.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA_50"], mode="lines", name="SMA 50"))
            fig_price.add_trace(go.Scatter(x=tech_df.index, y=tech_df["SMA_200"], mode="lines", name="SMA 200"))
            fig_price.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("## 📉 RSI (14)")
            rsi_plot_df = tech_df[["RSI_14"]].dropna()
            if not rsi_plot_df.empty:
                fig_rsi = px.line(rsi_plot_df, x=rsi_plot_df.index, y="RSI_14")
                fig_rsi.update_layout(height=300, xaxis_title="Date", yaxis_title="RSI")
                fig_rsi.add_hline(y=70, line_dash="dash")
                fig_rsi.add_hline(y=30, line_dash="dash")
                st.plotly_chart(fig_rsi, use_container_width=True)

            if not rolling_beta_series.empty:
                st.markdown("## 📈 Rolling Beta (60D)")
                rolling_beta_df = pd.DataFrame({"Rolling Beta": rolling_beta_series})
                fig_beta = px.line(rolling_beta_df, x=rolling_beta_df.index, y="Rolling Beta")
                fig_beta.update_layout(height=350, xaxis_title="Date", yaxis_title="Beta")
                fig_beta.add_hline(y=1.0, line_dash="dash")
                fig_beta.add_hline(y=0.0, line_dash="dot")
                st.plotly_chart(fig_beta, use_container_width=True)

# ============================================================
# PAGE 2: MULTI-ASSET COMPARISON
# ============================================================
elif page == "Multi-Asset Comparison":
    st.subheader("Multi-Asset Comparison")

    selected_class = st.multiselect(
        "Select asset classes",
        options=all_categories,
        default=all_categories[: min(2, len(all_categories))]
    )

    compare_df = universe_df[universe_df["category"].isin(selected_class)].copy() if selected_class else universe_df.head(0).copy()
    compare_df["label"] = compare_df["name"] + " (" + compare_df["ticker"] + ")"

    search_compare = st.text_input("Search comparison universe", placeholder="Type ticker or name")
    compare_options = search_labels(compare_df, search_compare)

    selected_labels = st.multiselect(
        "Select assets to compare",
        options=compare_options,
        default=compare_options[: min(8, len(compare_options))],
    )

    if not selected_labels:
        st.info("Please select at least one asset.")
    else:
        tickers = [label.split("(")[-1].replace(")", "").strip() for label in selected_labels]

        with st.spinner("Loading comparison data from MySQL..."):
            prices = load_price_history_from_db(tuple(tickers), start_date, end_date)

        comparison_prices = prices[tickers].dropna(how="all") if not prices.empty else pd.DataFrame()
        if comparison_prices.empty:
            st.warning("No data returned for the selected assets.")
        else:
            comparison_returns = compute_returns(comparison_prices)

            st.markdown("## 📈 Normalized Price Comparison")
            normalized_prices = comparison_prices.dropna()
            if not normalized_prices.empty:
                normalized = normalized_prices / normalized_prices.iloc[0] * 100
                fig_compare = px.line(normalized, x=normalized.index, y=normalized.columns)
                fig_compare.update_layout(height=500, xaxis_title="Date", yaxis_title="Normalized Value (Base = 100)")
                st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown("## 🔗 Correlation Matrix")
            if not comparison_returns.empty:
                corr = comparison_returns.corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
                fig_corr.update_layout(height=650)
                st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("## 📋 Performance Metrics")
            perf = performance_table(comparison_returns)
            st.dataframe(
                perf.style.format(
                    {
                        "Annual Return": "{:.2%}",
                        "Annual Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                        "Max Drawdown": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

# ============================================================
# PAGE 3: PORTFOLIO OPTIMIZER
# ============================================================
elif page == "Portfolio Optimizer":
    st.subheader("Portfolio Optimizer")
    st.write("Search for the best historical Sharpe portfolio from MySQL price history.")

    optimizer_universe = universe_df.copy()
    optimizer_universe["label"] = optimizer_universe["name"] + " (" + optimizer_universe["ticker"] + ")"

    search_optimizer = st.text_input("Search optimization universe", placeholder="Type ticker or name")
    optimizer_options = search_labels(optimizer_universe, search_optimizer)

    selected_labels = st.multiselect(
        "Select assets for optimization",
        options=optimizer_options,
        default=optimizer_options[: min(12, len(optimizer_options))],
    )

    col1, col2 = st.columns(2)
    with col1:
        n_portfolios = st.slider("Number of random portfolios", 1000, 20000, 5000, 1000)
    with col2:
        sharpe_target = st.number_input("Target Sharpe", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    if len(selected_labels) < 2:
        st.info("Please select at least two assets.")
    else:
        tickers = [label.split("(")[-1].replace(")", "").strip() for label in selected_labels]

        with st.spinner("Optimizing portfolio from MySQL data..."):
            prices = load_price_history_from_db(tuple(tickers), start_date, end_date)

        if prices.empty:
            st.warning("No data returned.")
        else:
            returns = compute_returns(prices).dropna()

            if returns.empty or returns.shape[1] < 2:
                st.warning("Not enough return data to optimize.")
            else:
                results_df, best_portfolio, best_weights = optimize_max_sharpe(returns_df=returns, n_portfolios=n_portfolios)

                if best_portfolio is None or best_weights.empty:
                    st.warning("Optimizer could not find a valid portfolio.")
                else:
                    best_portfolio_returns = portfolio_returns_from_weights(returns, best_weights)
                    growth_df = pd.DataFrame({"Optimized Portfolio": make_growth_curve(best_portfolio_returns)})

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Best Historical Sharpe", f"{best_portfolio['Sharpe']:.2f}")
                    m2.metric("Best Annual Return", f"{best_portfolio['Return']:.2%}")
                    m3.metric("Best Annual Volatility", f"{best_portfolio['Volatility']:.2%}")
                    m4.metric("Target Reached?", "Yes" if best_portfolio["Sharpe"] >= sharpe_target else "No")

                    st.markdown("## 🌐 Portfolio Simulation Cloud")
                    fig = px.scatter(results_df, x="Volatility", y="Return", color="Sharpe", title="Random Portfolio Simulation")
                    fig.update_layout(height=550)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("## 🧺 Best Portfolio Weights")
                    st.dataframe(best_weights.style.format({"Weight": "{:.2%}"}), use_container_width=True)

                    st.markdown("## 💰 Optimized Portfolio Growth")
                    fig_growth = px.line(growth_df, x=growth_df.index, y=growth_df.columns)
                    fig_growth.update_layout(height=450, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                    st.plotly_chart(fig_growth, use_container_width=True)

# ============================================================
# PAGE 4: CUSTOM PORTFOLIO BUILDER
# ============================================================
elif page == "Custom Portfolio Builder":
    st.subheader("Custom Portfolio Builder")
    st.write("Build your own portfolio with manual weights and choose Daily, Weekly, or Monthly rebalancing.")

    builder_universe = universe_df.copy()
    builder_universe["label"] = builder_universe["name"] + " (" + builder_universe["ticker"] + ")"
    search_builder = st.text_input("Search portfolio builder universe", placeholder="Type ticker or name")
    builder_options = search_labels(builder_universe, search_builder)

    selected_labels = st.multiselect(
        "Select assets for your custom portfolio",
        options=builder_options,
        default=builder_options[: min(8, len(builder_options))],
    )

    rebalance_frequency = st.selectbox("Rebalancing frequency", ["Daily", "Weekly", "Monthly"], index=2)
    weight_mode = st.radio("Weight mode", ["Custom Weights", "Equal Weight"], horizontal=True)

    if len(selected_labels) < 2:
        st.info("Please select at least two assets.")
    else:
        selected_tickers = [label.split("(")[-1].replace(")", "").strip() for label in selected_labels]
        default_weight = round(100 / len(selected_tickers), 2)

        st.markdown("## Set Portfolio Weights (%)")
        weight_inputs = {}
        cols = st.columns(3)

        if weight_mode == "Equal Weight":
            for i, ticker in enumerate(selected_tickers):
                with cols[i % 3]:
                    st.number_input(
                        f"{ticker} weight %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_weight),
                        step=1.0,
                        key=f"equal_weight_{ticker}",
                        disabled=True,
                    )
                weight_inputs[ticker] = default_weight
        else:
            for i, ticker in enumerate(selected_tickers):
                with cols[i % 3]:
                    weight_inputs[ticker] = st.number_input(
                        f"{ticker} weight %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_weight),
                        step=1.0,
                        key=f"custom_weight_{ticker}",
                    )

        normalized_weights_map = normalize_weight_inputs(weight_inputs)

        if not normalized_weights_map:
            st.warning("Please enter positive weights.")
        else:
            weights_df = pd.DataFrame(
                {"Ticker": list(normalized_weights_map.keys()), "Weight": list(normalized_weights_map.values())}
            ).sort_values("Weight", ascending=False)

            with st.spinner("Building custom portfolio from MySQL data..."):
                prices = load_price_history_from_db(tuple(selected_tickers + [benchmark_ticker]), start_date, end_date)

            portfolio_prices = prices[selected_tickers].dropna(how="any") if not prices.empty else pd.DataFrame()

            if portfolio_prices.empty:
                st.warning("No price data returned for the selected portfolio.")
            else:
                portfolio_output = backtest_custom_portfolio(
                    prices=portfolio_prices,
                    weights_map=normalized_weights_map,
                    rebalance_frequency=rebalance_frequency,
                )

                portfolio_returns = portfolio_output["returns"].dropna()
                if not portfolio_returns.empty:
                    st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), use_container_width=True)
                    growth_df = pd.DataFrame({"Custom Portfolio": make_growth_curve(portfolio_returns)})
                    fig_growth = px.line(growth_df, x=growth_df.index, y=growth_df.columns)
                    fig_growth.update_layout(height=500)
                    st.plotly_chart(fig_growth, use_container_width=True)

# ============================================================
# PAGE 5: ALPHA RESEARCH LAB
# ============================================================
elif page == "Alpha Research Lab":
    st.subheader("Alpha Research Lab")
    st.write("Research factor signals, rank assets cross-sectionally, and test forward-return predictiveness.")

    alpha_universe = universe_df.copy()
    alpha_universe["label"] = alpha_universe["name"] + " (" + alpha_universe["ticker"] + ")"
    search_alpha = st.text_input("Search alpha universe", placeholder="Type ticker or name")
    alpha_options = search_labels(alpha_universe, search_alpha)

    selected_labels = st.multiselect(
        "Select assets for alpha research",
        options=alpha_options,
        default=alpha_options[: min(12, len(alpha_options))],
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        signal_name = st.selectbox("Factor signal", ["Momentum", "Volatility Adjusted Momentum", "Mean Reversion", "Short-Term Reversal"], index=0)
    with col2:
        lookback = st.number_input("Signal lookback", min_value=5, max_value=252, value=60, step=5)
    with col3:
        forward_horizon = st.number_input("Forward return horizon", min_value=1, max_value=126, value=20, step=1)
    with col4:
        n_buckets = st.number_input("Signal buckets", min_value=3, max_value=10, value=5, step=1)

    if len(selected_labels) < 4:
        st.info("Please select at least four assets for cross-sectional alpha research.")
    else:
        tickers = [label.split("(")[-1].replace(")", "").strip() for label in selected_labels]

        with st.spinner("Running alpha research from MySQL data..."):
            prices = load_price_history_from_db(tuple(tickers), start_date, end_date)

        if prices.empty or prices.shape[1] < 4:
            st.warning("Not enough price data returned for alpha analysis.")
        else:
            signal_df = compute_factor_signal(prices, signal_name=signal_name, lookback=int(lookback))
            ranked_signal_df = cross_sectional_rank(signal_df)
            forward_returns_df = compute_forward_returns(prices, horizon=int(forward_horizon))
            daily_ic = compute_daily_ic(signal_df, forward_returns_df)
            hit_rate = compute_hit_rate(signal_df, forward_returns_df)
            spread_df = compute_quantile_spread(signal_df, forward_returns_df, n_buckets=int(n_buckets))

            avg_ic = daily_ic.mean()
            ic_std = daily_ic.std()
            ic_ir = np.nan if pd.isna(ic_std) or ic_std == 0 else avg_ic / ic_std

            m1, m2, m3 = st.columns(3)
            m1.metric("Average IC", "N/A" if pd.isna(avg_ic) else f"{avg_ic:.3f}")
            m2.metric("IC IR", "N/A" if pd.isna(ic_ir) else f"{ic_ir:.3f}")
            m3.metric("Hit Rate", "N/A" if pd.isna(hit_rate) else f"{hit_rate:.2%}")

            if not daily_ic.dropna().empty:
                ic_df = pd.DataFrame({"Daily IC": daily_ic})
                fig_ic = px.line(ic_df, x=ic_df.index, y="Daily IC")
                fig_ic.update_layout(height=350)
                fig_ic.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_ic, use_container_width=True)

            if not spread_df.empty and "Top-Bottom Spread" in spread_df.columns:
                spread_plot_df = pd.DataFrame({"Top-Bottom Spread": spread_df["Top-Bottom Spread"]})
                fig_spread = px.line(spread_plot_df, x=spread_plot_df.index, y="Top-Bottom Spread")
                fig_spread.update_layout(height=350)
                fig_spread.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_spread, use_container_width=True)

# ============================================================
# PAGE 6: SIGNAL DASHBOARD
# ============================================================
elif page == "Signal Dashboard":
    st.subheader("BUY / HOLD / SELL Signal Dashboard")
    st.write("Read latest signals directly from MySQL.")

    signal_universe = universe_df.copy()
    signal_universe["label"] = signal_universe["name"] + " (" + signal_universe["ticker"] + ")"

    search_signal = st.text_input("Search signal universe", placeholder="Type ticker or name")
    signal_options = search_labels(signal_universe, search_signal)

    selected_labels = st.multiselect(
        "Select assets for signal display",
        options=signal_options,
        default=signal_options[: min(12, len(signal_options))],
    )

    if len(selected_labels) == 0:
        st.info("Please select at least one asset.")
    else:
        tickers = tuple(label.split("(")[-1].replace(")", "").strip() for label in selected_labels)

        with st.spinner("Loading signals from MySQL..."):
            signal_table = load_latest_signal_table_from_db(tickers)

        if signal_table.empty:
            st.warning("No signals found in MySQL for the selected assets.")
        else:
            buy_count = int((signal_table["signal_label"] == "BUY").sum())
            hold_count = int((signal_table["signal_label"] == "HOLD").sum())
            sell_count = int((signal_table["signal_label"] == "SELL").sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("BUY", buy_count)
            m2.metric("HOLD", hold_count)
            m3.metric("SELL", sell_count)

            display_df = signal_table.rename(
                columns={
                    "ticker": "Ticker",
                    "signal_date": "Signal Date",
                    "signal_type": "Signal Type",
                    "signal_value": "Signal Value",
                    "signal_label": "Signal",
                    "short_ma": "Short MA",
                    "long_ma": "Long MA",
                    "momentum_window": "Momentum Window",
                }
            )

            st.dataframe(display_df.style.format({"Signal Value": "{:.4f}"}), use_container_width=True)

# ============================================================
# PAGE 7: QUANT STRATEGY BACKTESTER
# ============================================================
elif page == "Quant Strategy Backtester":
    st.subheader("Quant Strategy Backtester")
    st.write("Momentum + inverse-volatility weighting + volatility targeting + optional SPY regime filter")

    strategy_defaults = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "LQD"]
    strategy_universe = universe_df.copy()
    strategy_universe["label"] = strategy_universe["name"] + " (" + strategy_universe["ticker"] + ")"

    search_strategy = st.text_input("Search strategy universe", placeholder="Type ticker or name")
    strategy_options = search_labels(strategy_universe, search_strategy)

    default_labels = [label for label in strategy_options if label.split("(")[-1].replace(")", "").strip() in strategy_defaults]
    if not default_labels:
        default_labels = strategy_options[: min(6, len(strategy_options))]

    strategy_labels = st.multiselect("Select assets for the strategy", options=strategy_options, default=default_labels)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lookback = st.number_input("Momentum lookback", min_value=20, max_value=252, value=60, step=5)
    with c2:
        vol_window = st.number_input("Volatility window", min_value=10, max_value=126, value=20, step=5)
    with c3:
        top_n = st.number_input("Top N assets", min_value=1, max_value=15, value=3, step=1)
    with c4:
        target_vol = st.number_input("Target volatility", min_value=0.05, max_value=0.30, value=0.10, step=0.01)

    use_regime_filter = st.checkbox("Use SPY 200-day regime filter", value=True)

    if len(strategy_labels) < 2:
        st.info("Please select at least two assets for the strategy.")
    else:
        strategy_tickers = [label.split("(")[-1].replace(")", "").strip() for label in strategy_labels]

        if use_regime_filter and "SPY" not in strategy_tickers:
            strategy_tickers.append("SPY")

        with st.spinner("Running backtest from MySQL data..."):
            prices = load_price_history_from_db(tuple(strategy_tickers), start_date, end_date)

        if prices.empty:
            st.warning("No data returned for the selected strategy assets.")
        else:
            strategy_output = build_momentum_strategy(
                prices=prices,
                lookback=int(lookback),
                vol_window=int(vol_window),
                top_n=int(top_n),
                target_vol=float(target_vol),
                use_regime_filter=use_regime_filter,
            )

            strategy_returns = strategy_output["returns"].dropna()
            if strategy_returns.empty:
                st.warning("Strategy returns could not be computed.")
            else:
                cumulative = pd.DataFrame({"Strategy": (1 + strategy_returns).cumprod()})
                fig_cum = px.line(cumulative, x=cumulative.index, y=cumulative.columns)
                fig_cum.update_layout(height=500, xaxis_title="Date", yaxis_title="Growth of $1")
                st.plotly_chart(fig_cum, use_container_width=True)

                growth_df = pd.DataFrame({"Strategy": make_growth_curve(strategy_returns)})
                fig_growth = px.line(growth_df, x=growth_df.index, y=growth_df.columns)
                fig_growth.update_layout(height=500, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig_growth, use_container_width=True)

# ============================================================
# PAGE 8: ADVANCED SQL LAB
# ============================================================
elif page == "Advanced SQL Lab":
    st.subheader("Advanced SQL Lab")
    st.write("Run advanced MySQL analytics directly on your database.")

    sql_view = st.selectbox(
        "Select SQL analytics view",
        [
            "Top Momentum",
            "Worst Momentum",
            "Highest Volatility",
            "Latest Prices",
            "Signal Summary",
        ],
    )

    col1, col2 = st.columns(2)
    with col1:
        limit_n = st.number_input("Rows", min_value=5, max_value=100, value=25, step=5)
    with col2:
        lookback_days = st.number_input("Lookback days", min_value=5, max_value=252, value=20, step=5)

    if sql_view == "Top Momentum":
        df = sql_top_momentum(limit_n=int(limit_n), lookback_days=int(lookback_days))
        st.markdown("## Top Momentum Tickers")
        st.dataframe(df.style.format({"latest_price": "{:.2f}", "lookback_price": "{:.2f}", "momentum_return": "{:.2%}"}), use_container_width=True)

    elif sql_view == "Worst Momentum":
        df = sql_bottom_momentum(limit_n=int(limit_n), lookback_days=int(lookback_days))
        st.markdown("## Worst Momentum Tickers")
        st.dataframe(df.style.format({"latest_price": "{:.2f}", "lookback_price": "{:.2f}", "momentum_return": "{:.2%}"}), use_container_width=True)

    elif sql_view == "Highest Volatility":
        df = sql_highest_volatility(limit_n=int(limit_n), lookback_days=int(lookback_days))
        st.markdown("## Highest Volatility Tickers")
        st.dataframe(df.style.format({"annualized_volatility": "{:.2%}"}), use_container_width=True)

    elif sql_view == "Latest Prices":
        df = sql_latest_prices(limit_n=int(limit_n))
        st.markdown("## Latest Prices")
        st.dataframe(df.style.format({"close_price": "{:.2f}"}), use_container_width=True)

    elif sql_view == "Signal Summary":
        df = sql_signal_summary()
        st.markdown("## Latest Signal Counts")
        st.dataframe(df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"Built with Python, Streamlit, MySQL, Pandas, NumPy, and Plotly | © {AUTHOR_NAME}")
