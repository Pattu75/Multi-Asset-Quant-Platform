import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Zakariya Boutayeb | Multi-Asset Quant Platform",
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

# ============================================================
# CURATED DEFAULT UNIVERSE
# ============================================================
DEFAULT_UNIVERSE = {
    "Large Cap Stocks": {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "NVIDIA": "NVDA",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Berkshire Hathaway": "BRK-B",
        "JPMorgan": "JPM",
        "Johnson & Johnson": "JNJ",
        "Exxon Mobil": "XOM",
        "Visa": "V",
        "Mastercard": "MA",
        "Eli Lilly": "LLY",
        "Costco": "COST",
        "Netflix": "NFLX",
    },
    "Mid Cap Stocks": {
        "DraftKings": "DKNG",
        "Robinhood": "HOOD",
        "RPM International": "RPM",
        "Reliance Steel": "RS",
        "EMCOR": "EME",
        "Duolingo": "DUOL",
        "Cava Group": "CAVA",
    },
    "Small Cap Stocks": {
        "Cleanspark": "CLSK",
        "Astera Labs": "ALAB",
        "TransMedics": "TMDX",
        "Sirius XM": "SIRI",
        "Hims & Hers": "HIMS",
        "SoFi": "SOFI",
    },
    "ETFs": {
        "SPDR S&P 500 ETF": "SPY",
        "Invesco QQQ": "QQQ",
        "iShares Russell 2000 ETF": "IWM",
        "Vanguard Total Stock Market ETF": "VTI",
        "Vanguard FTSE Developed Markets ETF": "VEA",
        "Vanguard FTSE Emerging Markets ETF": "VWO",
        "Financial Select Sector SPDR": "XLF",
        "Technology Select Sector SPDR": "XLK",
        "Energy Select Sector SPDR": "XLE",
    },
    "Bonds": {
        "iShares 20+ Year Treasury Bond ETF": "TLT",
        "iShares 7-10 Year Treasury Bond ETF": "IEF",
        "iShares 1-3 Year Treasury Bond ETF": "SHY",
        "iShares Investment Grade Corporate Bond ETF": "LQD",
        "iShares High Yield Corporate Bond ETF": "HYG",
        "US Aggregate Bond ETF": "AGG",
        "TIPS Bond ETF": "TIP",
    },
    "Commodities": {
        "SPDR Gold Shares": "GLD",
        "iShares Silver Trust": "SLV",
        "United States Oil Fund": "USO",
        "Invesco DB Commodity Index": "DBC",
        "Abrdn Physical Palladium": "PALL",
    },
    "REITs": {
        "Vanguard Real Estate ETF": "VNQ",
        "Realty Income": "O",
        "Simon Property Group": "SPG",
        "Prologis": "PLD",
        "Digital Realty": "DLR",
        "Equinix": "EQIX",
    },
    "Indexes": {
        "S&P 500 Index": "^GSPC",
        "Nasdaq 100 Index": "^NDX",
        "Dow Jones Industrial Average": "^DJI",
        "Russell 2000 Index": "^RUT",
        "VIX": "^VIX",
    },
    "Crypto Proxies": {
        "Bitcoin ETF": "IBIT",
        "Ethereum ETF": "ETHA",
        "Coinbase": "COIN",
        "MicroStrategy": "MSTR",
    },
}

BENCHMARKS = {
    "S&P 500 ETF (SPY)": "SPY",
    "S&P 500 Index (^GSPC)": "^GSPC",
    "Nasdaq 100 ETF (QQQ)": "QQQ",
    "Russell 2000 ETF (IWM)": "IWM",
    "US Aggregate Bond ETF (AGG)": "AGG",
}

# ============================================================
# HELPERS: UNIVERSE
# ============================================================
def default_universe_df() -> pd.DataFrame:
    rows = []
    for category, assets in DEFAULT_UNIVERSE.items():
        for name, ticker in assets.items():
            rows.append(
                {
                    "category": category,
                    "name": name,
                    "ticker": ticker,
                }
            )
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["ticker"])
        .sort_values(["category", "name"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def build_large_ticker_universe() -> pd.DataFrame:
    universe = default_universe_df().copy()

    wiki_tables = [
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            0,
            "Large Cap Stocks",
            "Security",
            "Symbol",
        ),
        (
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            4,
            "Large Cap Stocks",
            "Company",
            "Ticker",
        ),
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            0,
            "Mid Cap Stocks",
            "Security",
            "Symbol",
        ),
        (
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            0,
            "Small Cap Stocks",
            "Company",
            "Symbol",
        ),
    ]

    extra_frames = []
    for url, table_idx, category, name_col, ticker_col in wiki_tables:
        try:
            table = pd.read_html(url)[table_idx]
            table = table[[name_col, ticker_col]].copy()
            table.columns = ["name", "ticker"]
            table["ticker"] = (
                table["ticker"]
                .astype(str)
                .str.replace(".", "-", regex=False)
                .str.strip()
                .str.upper()
            )
            table["name"] = table["name"].astype(str).str.strip()
            table["category"] = category
            extra_frames.append(table[["category", "name", "ticker"]])
        except Exception:
            continue

    if extra_frames:
        universe = pd.concat([universe] + extra_frames, ignore_index=True)

    universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    universe["name"] = universe["name"].astype(str).str.strip()
    universe["category"] = universe["category"].astype(str).str.strip()

    universe = universe.dropna(subset=["ticker", "name", "category"])
    universe = universe.drop_duplicates(subset=["ticker"], keep="first")
    universe = universe.sort_values(["category", "name"]).reset_index(drop=True)
    return universe


# ============================================================
# HELPERS: DATA
# ============================================================
@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date, end_date):
    tickers = [t for t in dict.fromkeys(tickers) if t]
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        if isinstance(data.columns, pd.Index) and "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            prices = pd.DataFrame()
    else:
        close_dict = {}
        for ticker in tickers:
            try:
                close_dict[ticker] = data[ticker]["Close"]
            except Exception:
                continue
        prices = pd.DataFrame(close_dict)

    return prices.dropna(how="all")


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


# ============================================================
# HELPERS: OPTIMIZER
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


# ============================================================
# HELPERS: SIGNALS
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


# ============================================================
# HELPERS: STRATEGY
# ============================================================
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
# TERMINAL / HEADER
# ============================================================
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

universe_df = build_large_ticker_universe()
all_categories = sorted(universe_df["category"].unique().tolist())
search_choices = [f"{row['ticker']} | {row['name']} | {row['category']}" for _, row in universe_df.iterrows()]

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
        "Signal Dashboard",
        "Quant Strategy Backtester",
    ],
)

st.sidebar.header("Global Settings")
start_date = st.sidebar.date_input("Start date", value=date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", value=date.today())
benchmark_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
benchmark_ticker = BENCHMARKS[benchmark_name]

st.sidebar.markdown("---")
st.sidebar.markdown("### Universe")
st.sidebar.metric("Tickers loaded", f"{len(universe_df):,}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Author")
st.sidebar.markdown(AUTHOR_NAME)
st.sidebar.markdown(AUTHOR_TAG)
st.sidebar.markdown("GitHub: Pattu75")

if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()

# ============================================================
# PAGE 0: TERMINAL
# ============================================================
if page == "Research Terminal":
    st.subheader("Market Research Terminal")
    st.caption("Multi-Asset Market Intelligence & Quantitative Analytics")

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

    col_a, col_b = st.columns([1.2, 1.8])

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

        with st.spinner("Downloading asset data..."):
            terminal_prices = download_prices([selected_ticker, benchmark_ticker], start_date, end_date)

        if terminal_prices.empty or selected_ticker not in terminal_prices.columns:
            st.warning("No data returned for this asset.")
        else:
            asset_prices = terminal_prices[[selected_ticker]].dropna()
            asset_returns = compute_returns(asset_prices)[selected_ticker]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Ticker", selected_ticker)
            m2.metric("Annual Return", f"{annualized_return(asset_returns):.2%}")
            m3.metric("Annual Volatility", f"{annualized_volatility(asset_returns):.2%}")
            m4.metric("Sharpe Ratio", f"{sharpe_ratio(asset_returns):.2f}")

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
    class_df["label"] = class_df["name"] + " (" + class_df["ticker"] + ")"
    selected_label = st.selectbox("Select asset", class_df["label"].tolist())
    ticker = class_df.loc[class_df["label"] == selected_label, "ticker"].iloc[0]

    with st.spinner("Downloading market data..."):
        prices = download_prices([ticker, benchmark_ticker], start_date, end_date)

    if prices.empty or ticker not in prices.columns:
        st.warning("No data returned for the selected ticker and date range.")
    else:
        asset_prices = prices[[ticker]].dropna()
        asset_returns = compute_returns(asset_prices)[ticker]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", ticker)
        c2.metric("Annual Return", f"{annualized_return(asset_returns):.2%}")
        c3.metric("Annual Volatility", f"{annualized_volatility(asset_returns):.2%}")
        c4.metric("Sharpe Ratio", f"{sharpe_ratio(asset_returns):.2f}")

        st.markdown("## 📊 Price Chart")
        fig_price = px.line(asset_prices, x=asset_prices.index, y=ticker)
        fig_price.update_layout(height=500, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

        if benchmark_ticker in prices.columns:
            st.markdown("## 📈 Normalized Performance vs Benchmark")
            combined = prices[[ticker, benchmark_ticker]].dropna()
            if not combined.empty:
                normalized = combined / combined.iloc[0] * 100
                fig_norm = px.line(normalized, x=normalized.index, y=normalized.columns)
                fig_norm.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Normalized Value (Base = 100)"
                )
                st.plotly_chart(fig_norm, use_container_width=True)

        st.markdown("## 📥 Download Data")
        st.download_button(
            label="Download price data as CSV",
            data=asset_prices.to_csv().encode("utf-8"),
            file_name=f"{ticker}_prices.csv",
            mime="text/csv",
        )

# ============================================================
# PAGE 2: MULTI-ASSET COMPARISON
# ============================================================
elif page == "Multi-Asset Comparison":
    st.subheader("Multi-Asset Comparison")

    selected_class = st.multiselect(
        "Select asset classes",
        options=all_categories,
        default=["ETFs", "Bonds"]
    )

    compare_df = universe_df[universe_df["category"].isin(selected_class)].copy() if selected_class else universe_df.head(0).copy()
    compare_df["label"] = compare_df["name"] + " (" + compare_df["ticker"] + ")"

    selected_labels = st.multiselect(
        "Select assets to compare",
        options=compare_df["label"].tolist(),
        default=compare_df["label"].tolist()[:4],
    )

    if not selected_labels:
        st.info("Please select at least one asset.")
    else:
        selected_rows = compare_df[compare_df["label"].isin(selected_labels)]
        tickers = selected_rows["ticker"].tolist()

        with st.spinner("Downloading comparison data..."):
            prices = download_prices(tickers + [benchmark_ticker], start_date, end_date)

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
                fig_compare.update_layout(
                    height=500,
                    xaxis_title="Date",
                    yaxis_title="Normalized Value (Base = 100)"
                )
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

            st.download_button(
                label="Download comparison data as CSV",
                data=comparison_prices.to_csv().encode("utf-8"),
                file_name="multi_asset_comparison.csv",
                mime="text/csv",
            )

# ============================================================
# PAGE 3: PORTFOLIO OPTIMIZER
# ============================================================
elif page == "Portfolio Optimizer":
    st.subheader("Portfolio Optimizer")
    st.write("Search for the best historical Sharpe portfolio from your selected assets. This is research-oriented and not a guarantee of future Sharpe > 2.")

    optimizer_universe = universe_df[
        universe_df["category"].isin(
            ["Large Cap Stocks", "Mid Cap Stocks", "Small Cap Stocks", "ETFs", "Bonds", "Commodities", "REITs", "Crypto Proxies"]
        )
    ].copy()
    optimizer_universe["label"] = optimizer_universe["name"] + " (" + optimizer_universe["ticker"] + ")"

    selected_labels = st.multiselect(
        "Select assets for optimization",
        options=optimizer_universe["label"].tolist(),
        default=optimizer_universe["label"].tolist()[:8],
    )

    col1, col2 = st.columns(2)
    with col1:
        n_portfolios = st.slider("Number of random portfolios", 1000, 20000, 5000, 1000)
    with col2:
        sharpe_target = st.number_input("Target Sharpe", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    if len(selected_labels) < 2:
        st.info("Please select at least two assets.")
    else:
        selected_rows = optimizer_universe[optimizer_universe["label"].isin(selected_labels)]
        tickers = selected_rows["ticker"].tolist()

        with st.spinner("Optimizing portfolio..."):
            prices = download_prices(tickers, start_date, end_date)

        if prices.empty:
            st.warning("No data returned.")
        else:
            returns = compute_returns(prices).dropna()

            if returns.empty or returns.shape[1] < 2:
                st.warning("Not enough return data to optimize.")
            else:
                results_df, best_portfolio, best_weights = optimize_max_sharpe(
                    returns_df=returns,
                    n_portfolios=n_portfolios,
                )

                if best_portfolio is None or best_weights.empty:
                    st.warning("Optimizer could not find a valid portfolio.")
                else:
                    best_portfolio_returns = portfolio_returns_from_weights(returns, best_weights)
                    growth_df = pd.DataFrame(
                        {
                            "Optimized Portfolio": make_growth_curve(best_portfolio_returns)
                        }
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Best Historical Sharpe", f"{best_portfolio['Sharpe']:.2f}")
                    m2.metric("Best Annual Return", f"{best_portfolio['Return']:.2%}")
                    m3.metric("Best Annual Volatility", f"{best_portfolio['Volatility']:.2%}")
                    m4.metric("Target Reached?", "Yes" if best_portfolio["Sharpe"] >= sharpe_target else "No")

                    st.markdown("## 🌐 Portfolio Simulation Cloud")
                    fig = px.scatter(
                        results_df,
                        x="Volatility",
                        y="Return",
                        color="Sharpe",
                        title="Random Portfolio Simulation",
                    )
                    fig.update_layout(height=550)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("## 🧺 Best Portfolio Weights")
                    st.dataframe(
                        best_weights.style.format({"Weight": "{:.2%}"}),
                        use_container_width=True,
                    )

                    st.markdown("## 💰 Optimized Portfolio Growth")
                    fig_growth = px.line(growth_df, x=growth_df.index, y=growth_df.columns)
                    fig_growth.update_layout(height=450, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                    st.plotly_chart(fig_growth, use_container_width=True)

                    st.download_button(
                        label="Download optimized weights as CSV",
                        data=best_weights.to_csv(index=False).encode("utf-8"),
                        file_name="optimized_portfolio_weights.csv",
                        mime="text/csv",
                    )

# ============================================================
# PAGE 4: SIGNAL DASHBOARD
# ============================================================
elif page == "Signal Dashboard":
    st.subheader("BUY / HOLD / SELL Signal Dashboard")
    st.write("Signals are based on moving-average trend and momentum. Use this for research and screening, not as guaranteed investment advice.")

    signal_universe = universe_df[
        universe_df["category"].isin(
            ["Large Cap Stocks", "Mid Cap Stocks", "Small Cap Stocks", "ETFs", "Bonds", "Commodities", "REITs", "Crypto Proxies"]
        )
    ].copy()
    signal_universe["label"] = signal_universe["name"] + " (" + signal_universe["ticker"] + ")"

    selected_labels = st.multiselect(
        "Select assets for signal generation",
        options=signal_universe["label"].tolist(),
        default=signal_universe["label"].tolist()[:10],
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        short_ma = st.number_input("Short moving average", min_value=10, max_value=100, value=50, step=5)
    with c2:
        long_ma = st.number_input("Long moving average", min_value=100, max_value=300, value=200, step=10)
    with c3:
        momentum_window = st.number_input("Momentum window", min_value=20, max_value=180, value=60, step=5)

    if len(selected_labels) == 0:
        st.info("Please select at least one asset.")
    else:
        selected_rows = signal_universe[signal_universe["label"].isin(selected_labels)]
        tickers = selected_rows["ticker"].tolist()

        with st.spinner("Generating signals..."):
            prices = download_prices(tickers, start_date, end_date)

        if prices.empty:
            st.warning("No data returned.")
        else:
            signal_table = latest_signal_table(
                prices,
                short_ma=int(short_ma),
                long_ma=int(long_ma),
                momentum_window=int(momentum_window),
            )

            if signal_table.empty:
                st.warning("Not enough data to generate signals.")
            else:
                buy_count = int((signal_table["Signal"] == "BUY").sum())
                hold_count = int((signal_table["Signal"] == "HOLD").sum())
                sell_count = int((signal_table["Signal"] == "SELL").sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("BUY", buy_count)
                m2.metric("HOLD", hold_count)
                m3.metric("SELL", sell_count)

                st.markdown("## 📋 Latest Signal Table")
                st.dataframe(
                    signal_table.style.format(
                        {
                            "Last Price": "{:.2f}",
                            "60D Return": "{:.2%}",
                            "20D Vol": "{:.2%}",
                        }
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    label="Download signal table as CSV",
                    data=signal_table.to_csv(index=False).encode("utf-8"),
                    file_name="signal_dashboard.csv",
                    mime="text/csv",
                )

# ============================================================
# PAGE 5: QUANT STRATEGY BACKTESTER
# ============================================================
elif page == "Quant Strategy Backtester":
    st.subheader("Quant Strategy Backtester")
    st.write("Momentum + inverse-volatility weighting + volatility targeting + optional SPY regime filter")

    strategy_defaults = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "LQD"]
    strategy_universe = universe_df[
        universe_df["category"].isin(["ETFs", "Bonds", "Commodities", "REITs", "Indexes", "Crypto Proxies"])
    ].copy()
    strategy_universe["label"] = strategy_universe["name"] + " (" + strategy_universe["ticker"] + ")"

    default_labels = strategy_universe[strategy_universe["ticker"].isin(strategy_defaults)]["label"].tolist()

    strategy_labels = st.multiselect(
        "Select assets for the strategy",
        options=strategy_universe["label"].tolist(),
        default=default_labels,
    )

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
        selected_rows = strategy_universe[strategy_universe["label"].isin(strategy_labels)]
        strategy_tickers = selected_rows["ticker"].tolist()

        if use_regime_filter and "SPY" not in strategy_tickers:
            strategy_tickers.append("SPY")

        with st.spinner("Running backtest..."):
            prices = download_prices(strategy_tickers, start_date, end_date)

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
            benchmark_prices = download_prices([benchmark_ticker], start_date, end_date)

            if benchmark_prices.empty or strategy_returns.empty:
                st.warning("Unable to compute strategy or benchmark returns.")
            else:
                benchmark_returns = compute_returns(benchmark_prices).iloc[:, 0]
                common_index = strategy_returns.index.intersection(benchmark_returns.index)
                strategy_returns = strategy_returns.loc[common_index]
                benchmark_returns = benchmark_returns.loc[common_index]

                if len(common_index) == 0:
                    st.warning("No overlapping return dates between strategy and benchmark.")
                else:
                    cumulative = pd.DataFrame(
                        {
                            "Strategy": (1 + strategy_returns).cumprod(),
                            benchmark_ticker: (1 + benchmark_returns).cumprod(),
                        }
                    )

                    growth_df = pd.DataFrame(
                        {
                            "Strategy": make_growth_curve(strategy_returns),
                            benchmark_ticker: make_growth_curve(benchmark_returns),
                        }
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Strategy Return", f"{annualized_return(strategy_returns):.2%}")
                    m2.metric("Strategy Volatility", f"{annualized_volatility(strategy_returns):.2%}")
                    m3.metric("Strategy Sharpe", f"{sharpe_ratio(strategy_returns):.2f}")
                    m4.metric("Strategy Max Drawdown", f"{max_drawdown(strategy_returns):.2%}")

                    st.markdown("## 📈 Strategy vs Benchmark")
                    fig_cum = px.line(cumulative, x=cumulative.index, y=cumulative.columns)
                    fig_cum.update_layout(height=500, xaxis_title="Date", yaxis_title="Growth of $1")
                    st.plotly_chart(fig_cum, use_container_width=True)

                    st.markdown("## 💰 Portfolio Growth of $10,000")
                    fig_growth = px.line(growth_df, x=growth_df.index, y=growth_df.columns)
                    fig_growth.update_layout(height=500, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                    st.plotly_chart(fig_growth, use_container_width=True)

                    st.markdown("## 📉 Drawdown")
                    drawdown_df = pd.DataFrame({"Drawdown": strategy_output["drawdown"].reindex(common_index)})
                    fig_dd = px.line(drawdown_df, x=drawdown_df.index, y="Drawdown")
                    fig_dd.update_layout(height=400, xaxis_title="Date", yaxis_title="Drawdown")
                    st.plotly_chart(fig_dd, use_container_width=True)

                    st.markdown("## 📊 Rolling Sharpe (60 Days)")
                    rolling_sharpe_df = pd.DataFrame(
                        {"Rolling Sharpe": strategy_output["rolling_sharpe"].reindex(common_index)}
                    )
                    fig_rs = px.line(rolling_sharpe_df, x=rolling_sharpe_df.index, y="Rolling Sharpe")
                    fig_rs.update_layout(height=400, xaxis_title="Date", yaxis_title="Rolling Sharpe")
                    st.plotly_chart(fig_rs, use_container_width=True)

                    st.markdown("## 🧺 Current Portfolio Weights")
                    weights_df = strategy_output["weights"]
                    if not weights_df.empty:
                        latest_weights = weights_df.iloc[-1]
                        latest_weights = latest_weights[latest_weights != 0].sort_values(ascending=False)
                        if latest_weights.empty:
                            st.info("No active positions on the latest date.")
                        else:
                            fig_weights = px.bar(
                                x=latest_weights.index,
                                y=latest_weights.values,
                                labels={"x": "Asset", "y": "Weight"}
                            )
                            fig_weights.update_layout(height=450)
                            st.plotly_chart(fig_weights, use_container_width=True)

                    st.markdown("## 📋 Strategy Return Series")
                    strategy_df = pd.DataFrame({"Strategy Returns": strategy_returns})
                    st.dataframe(strategy_df.tail(20), use_container_width=True)

                    st.download_button(
                        label="Download strategy returns as CSV",
                        data=strategy_df.to_csv().encode("utf-8"),
                        file_name="strategy_returns.csv",
                        mime="text/csv",
                    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(f"Built with Python, Streamlit, yFinance, Pandas, NumPy, and Plotly | © {AUTHOR_NAME}")
