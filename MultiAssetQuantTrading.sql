CREATE DATABASE IF NOT EXISTS Quant_Platform;
USE Quant_Platform;

-- =========================================================
-- 1. TICKERS MASTER
-- =========================================================
CREATE TABLE IF NOT EXISTS tickers (
    ticker_id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    asset_name VARCHAR(255),
    asset_class VARCHAR(100),
    exchange_name VARCHAR(100),
    sector VARCHAR(150),
    industry VARCHAR(150),
    currency VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE INDEX idx_tickers_asset_class ON tickers(asset_class);

CREATE INDEX idx_tickers_is_active ON tickers(is_active);

-- =========================================================
-- 2. DAILY PRICE HISTORY
-- =========================================================
CREATE TABLE IF NOT EXISTS price_history (
    price_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT NOT NULL,
    trade_date DATE NOT NULL,
    open_price DECIMAL(18,6),
    high_price DECIMAL(18,6),
    low_price DECIMAL(18,6),
    close_price DECIMAL(18,6),
    adj_close_price DECIMAL(18,6),
    volume BIGINT,
    data_source VARCHAR(50) DEFAULT 'Yahoo Finance',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_price_ticker FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id) ON DELETE CASCADE,
    CONSTRAINT uq_price_ticker_date UNIQUE (ticker_id, trade_date)
) ENGINE=InnoDB;

CREATE INDEX idx_price_ticker_date ON price_history(ticker_id, trade_date);

CREATE INDEX idx_price_trade_date ON price_history(trade_date);

-- =========================================================
-- 3. SIGNALS TABLE
-- =========================================================
CREATE TABLE IF NOT EXISTS signals (
    signal_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ticker_id INT NOT NULL,
    signal_date DATE NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    signal_value DECIMAL(18,8),
    signal_label VARCHAR(20) NOT NULL,
    short_ma INT,
    long_ma INT,
    momentum_window INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_signal_ticker FOREIGN KEY (ticker_id) REFERENCES tickers(ticker_id) ON DELETE CASCADE,
    CONSTRAINT uq_signal_ticker_date_type UNIQUE (ticker_id, signal_date, signal_type)
) ENGINE=InnoDB;

CREATE INDEX idx_signal_ticker_date ON signals(ticker_id, signal_date);

CREATE INDEX idx_signal_type_date ON signals(signal_type, signal_date);

-- =========================================================
-- ADVANCED ANALYTICS QUERIES
-- =========================================================

-- =========================================================
-- A. LATEST DAILY SNAPSHOT
-- =========================================================
WITH latest_prices AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price, p.volume,
        ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY p.trade_date DESC) AS rn
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
)
SELECT ticker, asset_name, asset_class, trade_date, close_price, volume FROM latest_prices
WHERE rn = 1
ORDER BY ticker;

-- =========================================================
-- B. MOST ACTIVE TICKERS
-- =========================================================
WITH ranked AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price, p.volume,
        ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY p.trade_date DESC) AS rn
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
)
SELECT ticker, asset_name, asset_class, trade_date, close_price, volume
FROM ranked
WHERE rn = 1
ORDER BY volume DESC LIMIT 25;

-- =========================================================
-- C. TOP GAINERS (1-DAY)
-- =========================================================
WITH px AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price,
        LAG(COALESCE(p.adj_close_price, p.close_price))
            OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS prev_close
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
),
latest AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
    FROM px
)
SELECT ticker, asset_name, asset_class, trade_date, close_price, prev_close,
    ROUND(((close_price / prev_close) - 1) * 100, 2) AS pct_change_1d
FROM latest
WHERE rn = 1 AND prev_close IS NOT NULL
ORDER BY pct_change_1d DESC LIMIT 25;

-- =========================================================
-- D. TOP LOSERS (1-DAY)
-- =========================================================
WITH px AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price,
        LAG(COALESCE(p.adj_close_price, p.close_price))
            OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS prev_close
    FROM price_history p
    JOIN tickers t
        ON p.ticker_id = t.ticker_id
),
latest AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
    FROM px
)
SELECT ticker, asset_name, asset_class, trade_date, close_price, prev_close,
    ROUND(((close_price / prev_close) - 1) * 100, 2) AS pct_change_1d
FROM latest
WHERE rn = 1 AND prev_close IS NOT NULL
ORDER BY pct_change_1d ASC LIMIT 25;

-- =========================================================
-- E. TOP 20-DAY MOMENTUM
-- =========================================================
WITH px AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price,
        LAG(COALESCE(p.adj_close_price, p.close_price), 20)
            OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS close_20d_ago
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
),
latest AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
    FROM px
)
SELECT
    ticker, asset_name, asset_class, trade_date, close_price, close_20d_ago,
    ROUND(((close_price / close_20d_ago) - 1) * 100, 2) AS momentum_20d_pct
FROM latest
WHERE rn = 1 AND close_20d_ago IS NOT NULL
ORDER BY momentum_20d_pct DESC LIMIT 25;

-- =========================================================
-- F. 52-WEEK GAINERS
-- =========================================================
WITH hist AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
),
latest AS (
    SELECT ticker, asset_name, asset_class, trade_date, close_price
    FROM (
        SELECT
            h.*,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
        FROM hist h
    ) x
    WHERE rn = 1
),
range_52w AS (
    SELECT l.ticker, MIN(h.close_price) AS low_52w, MAX(h.close_price) AS high_52w
    FROM latest l
    JOIN hist h ON l.ticker = h.ticker
     AND h.trade_date BETWEEN DATE_SUB(l.trade_date, INTERVAL 365 DAY) AND l.trade_date
    GROUP BY l.ticker
)
SELECT l.ticker, l.asset_name, l.asset_class, l.trade_date, l.close_price, r.low_52w, r.high_52w,
    ROUND(((l.close_price / r.low_52w) - 1) * 100, 2) AS gain_from_52w_low_pct
FROM latest l
JOIN range_52w r ON l.ticker = r.ticker
ORDER BY gain_from_52w_low_pct DESC LIMIT 25;

-- =========================================================
-- G. WORST DRAWDOWN FROM 52-WEEK HIGH
-- =========================================================
WITH hist AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
),
latest AS (
    SELECT *
    FROM (
        SELECT
            h.*,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
        FROM hist h
    ) x
    WHERE rn = 1
),
range_52w AS (
    SELECT l.ticker, MAX(h.close_price) AS high_52w
    FROM latest l
    JOIN hist h ON l.ticker = h.ticker
     AND h.trade_date BETWEEN DATE_SUB(l.trade_date, INTERVAL 365 DAY) AND l.trade_date
    GROUP BY l.ticker
)
SELECT l.ticker, l.asset_name, l.asset_class, l.trade_date, l.close_price, r.high_52w,
    ROUND(((l.close_price / r.high_52w) - 1) * 100, 2) AS drawdown_from_52w_high_pct
FROM latest l
JOIN range_52w r ON l.ticker = r.ticker
ORDER BY drawdown_from_52w_high_pct ASC
LIMIT 25;

-- =========================================================
-- H. STRONGEST RECENT VOLUME + PRICE MOVE
-- =========================================================
WITH px AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date,
        COALESCE(p.adj_close_price, p.close_price) AS close_price, p.volume,
        LAG(COALESCE(p.adj_close_price, p.close_price))
            OVER (PARTITION BY t.ticker ORDER BY p.trade_date) AS prev_close,
        AVG(p.volume) OVER (
            PARTITION BY t.ticker
            ORDER BY p.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS avg_vol_20d,
        ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY p.trade_date DESC) AS rn
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
)
SELECT ticker, asset_name, asset_class, trade_date, volume, avg_vol_20d,
    ROUND(((close_price / prev_close) - 1) * 100, 2) AS pct_change_1d,
    ROUND(volume / NULLIF(avg_vol_20d, 0), 2) AS volume_spike_ratio
FROM px
WHERE rn = 1 AND prev_close IS NOT NULL AND avg_vol_20d IS NOT NULL
ORDER BY volume_spike_ratio DESC, pct_change_1d DESC LIMIT 25;

-- =========================================================
-- I. UNUSUAL VOLUME / TRENDING PROXY
--    Trending = latest volume / 20-day average volume
-- =========================================================
WITH vol_base AS (
    SELECT t.ticker, t.asset_name, t.asset_class, p.trade_date, p.volume,
        AVG(p.volume) OVER (
            PARTITION BY t.ticker
            ORDER BY p.trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS avg_vol_20d,
        ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY p.trade_date DESC) AS rn
    FROM price_history p
    JOIN tickers t ON p.ticker_id = t.ticker_id
)
SELECT ticker, asset_name, asset_class, trade_date, volume, avg_vol_20d,
    ROUND(volume / NULLIF(avg_vol_20d, 0), 2) AS volume_spike_ratio
FROM vol_base
WHERE rn = 1
ORDER BY volume_spike_ratio DESC LIMIT 25;

-- =========================================================
-- J. CURRENT BUY SIGNALS
-- =========================================================
WITH latest_signals AS (
    SELECT t.ticker, t.asset_name, t.asset_class, s.signal_date, s.signal_label, s.signal_value,
        ROW_NUMBER() OVER (PARTITION BY t.ticker ORDER BY s.signal_date DESC) AS rn
    FROM signals s
    JOIN tickers t ON s.ticker_id = t.ticker_id
    WHERE s.signal_type = 'MA_MOMENTUM'
)
SELECT ticker, asset_name, asset_class, signal_date, signal_label, signal_value FROM latest_signals
WHERE rn = 1 AND signal_label = 'BUY'
ORDER BY signal_value DESC LIMIT 25;