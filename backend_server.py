from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
from datetime import datetime, timedelta
import warnings
import requests
import time
import os
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
warnings.filterwarnings('ignore')
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
# Removed TensorFlow and PyTorch imports - using scikit-learn alternatives
import matplotlib.cm as cm
from bs4.element import Tag

app = Flask(__name__)
CORS(app)

# Configuration
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds
THREAD_POOL_SIZE = 10
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Thread-safe cache
cache_lock = Lock()

# Model configurations
MODEL_CONFIGS = {
    1: {'name': 'XGBoost Quantile Regression', 'description': 'Low vol, precise'},
    2: {'name': 'Random Forest Bootstrap', 'description': 'Balanced'},
    3: {'name': 'Neural Network Conformal', 'description': 'High vol, complex'},
    4: {'name': 'Extra Trees Bootstrap', 'description': 'High vol, aggressive'},
    5: {'name': 'AdaBoost Conformal', 'description': 'High vol, adaptive'},
    6: {'name': 'Bayesian Ridge Conformal', 'description': 'Low vol, conservative'},
    7: {'name': 'Support Vector Regression', 'description': 'Balanced, stable'},
    8: {'name': 'Gradient Boosting Conformal', 'description': 'High vol, adaptive'},
    9: {'name': 'Elastic Net Conformal', 'description': 'Low vol, regularized'},
    10: {'name': 'Transformer', 'description': 'Deep learning, sequence modeling'}
}

# Index Wikipedia URLs
INDEX_URLS = {
    'SPY': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    'DOW': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
    'NASDAQ': 'https://en.wikipedia.org/wiki/Nasdaq-100',
    'SP400': 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
    'SPLV': '',  # Will be hardcoded later
    'SPHB': ''   # Will be hardcoded later
}

INDEX_ETF_TICKERS = {
    'NASDAQ': 'QQQ',
    'SPY': 'SPY',
    'DOW': 'DIA',
    'SP400': 'MDY',
}

def get_cached_data(cache_key):
    """Get data from cache if it exists and is not expired"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    with cache_lock:
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < CACHE_DURATION:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
    return None

def save_to_cache(cache_key, data):
    """Save data to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    with cache_lock:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")

def scrape_index_constituents(index_name, force_refresh=False):
    # Hardcoded SPHB tickers (since not available on Wikipedia)
    SPHB_TICKERS = [
        "MCHP", "SMCI", "MU", "ON", "NVDA", "AMD", "MPWR", "PLTR", "AVGO", "LRCX", "WDC", "DELL", "TER", "KLAC", "AMAT", "ORCL",
        "NXPI", "ANET", "INTC", "ADI", "STX", "HPE", "JBL", "SWKS", "CRWD", "QCOM", "TXN", "SNPS", "ZBRA", "APH", "CDNS", "TRMB",
        "KEYS", "NTAP", "NOW", "AAPL", "PANW",
        "GEV", "UAL", "DAL", "ETN", "URI", "PH", "EMR", "HWM", "GNRC", "PWR", "SWK", "LUV", "HUBB", "BA", "JCI", "GE", "ROK", "DAY",
        "TSLA", "CCL", "RCL", "NCLH", "CZR", "MGM", "DASH", "RL", "AMZN", "WSM", "EXPE", "ABNB", "DECK",
        "KKR", "SYF", "APO", "BX", "IVZ", "GS", "COF", "MS", "C", "AXP", "KEY", "CPAY",
        "ALB", "FCX", "NUE", "DD", "STLD",
        "MRNA", "CRL", "ALGN", "GEHC", "ISRG",
        "VST", "CEG", "NRG",
        "APA", "DVN", "HAL", "TPL",
        "WBD", "META",
        "EL"
    ]
    # Hardcoded SPLV tickers (since not available on Wikipedia)
    SPLV_TICKERS = [
        "EVRG", "ATO", "CMS", "WEC", "SO", "DUK", "PNW", "PPL", "LNT", "NI", "AEP", "DTE", "AEE", "EXC", "XEL", "ED", "FE", "PEG", "CNP",
        "CME", "ICE", "MMC", "WTW", "JKHY", "BRK/B", "L", "SPGI", "MA", "V", "FDS", "AON", "CB", "BRO", "CBOE", "AFL", "AJG",
        "KO", "CHD", "PG", "PEP", "KDP", "KMB", "MO", "SYY", "CL", "MDLZ", "TSN", "MKC", "COST", "GIS", "CLX",
        "LHX", "RSG", "ADP", "GD", "ITW", "VLTO", "UNP", "WM", "BR", "OTIS", "VRSK", "GWW", "ROL", "PAYX",
        "O", "VICI", "REG", "WELL", "AVB", "FRT", "CPT", "MAA", "VTR", "UDR", "EQR", "INVH",
        "HOLX", "JNJ", "COR", "MDT", "BDX", "ABT", "CAH", "SYK", "STE", "BSX", "DGX",
        "TJX", "ORLY", "AZO", "MCD", "YUM",
        "LIN", "ECL", "AVY",
        "MSI", "VRSN", "ROP",
        "VZ",
    ]
    if index_name == 'SPHB':
        return SPHB_TICKERS
    if index_name == 'SPLV':
        return SPLV_TICKERS
    cache_key = f"constituents_{index_name}"
    if not force_refresh:
        cached_data = get_cached_data(cache_key)
        if cached_data:
            return cached_data
    url = INDEX_URLS.get(index_name)
    if not url:
        return []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(response.content, 'html.parser')
        blacklist = {'THE', 'AND', 'FOR', 'INC', 'LTD', 'CORP', 'CO', 'LLC', 'ETF', 'PLC', 'GROUP', 'CLASS', 'FUND', 'TRUST', 'HOLDINGS', 'COMPANY', 'CORPORATION', 'LIMITED', 'NYSE', 'NASDAQ', 'NASDA'}
        tables = soup.find_all('table', class_='wikitable')
        best_tickers = []
        best_table = -1
        best_col = -1
        best_score = 0
        for table_idx, table in enumerate(tables):
            if not isinstance(table, Tag):
                continue
            rows = table.find_all('tr') if hasattr(table, 'find_all') else []
            if len(rows) < 2:
                continue
            header_cells = rows[0].find_all(['td', 'th']) if isinstance(rows[0], Tag) else []
            num_cols = len(header_cells)
            for i, row in enumerate(rows[:6]):
                if not isinstance(row, Tag):
                    continue
                cells = row.find_all(['td', 'th']) if hasattr(row, 'find_all') else []
                print(f"  Row {i}: {[cell.get_text(strip=True) if isinstance(cell, Tag) else str(cell) for cell in cells]}")
            for col_idx in range(num_cols):
                col_tickers = []
                for row in rows[1:]:
                    if not isinstance(row, Tag):
                        continue
                    cells = row.find_all(['td', 'th']) if hasattr(row, 'find_all') else []
                    if len(cells) > col_idx:
                        cell = cells[col_idx]
                        if isinstance(cell, Tag):
                            a = cell.find('a') if hasattr(cell, 'find') else None
                            cell_text = a.get_text(strip=True) if a and hasattr(a, 'get_text') else cell.get_text(strip=True) if hasattr(cell, 'get_text') else str(cell)
                        else:
                            cell_text = str(cell)
                        candidate = re.match(r'^([A-Z]{1,5})$', cell_text.replace('.', '').replace(' ', ''))
                        ticker = candidate.group(1) if candidate else None
                        if ticker and ticker not in blacklist:
                            col_tickers.append(ticker)
                header_text = header_cells[col_idx].get_text(strip=True).lower() if isinstance(header_cells[col_idx], Tag) else str(header_cells[col_idx]).lower()
                print(f"    Table {table_idx+1} Col {col_idx+1} header: '{header_text}' first 10 values: {col_tickers[:10]}")
                unique_tickers = set(col_tickers)
                score = len(unique_tickers)
                if header_text in ['symbol', 'ticker']:
                    score += 1000  # strong preference for 'symbol' or 'ticker' header
                if score > best_score and len(unique_tickers) > 5:
                    best_score = score
                    best_tickers = list(unique_tickers)
                    best_table = table_idx + 1
                    best_col = col_idx + 1
        if len(best_tickers) >= 5:
            print(f"[scrape_index_constituents] {index_name}: Used table #{best_table}, column #{best_col}, found {len(best_tickers)} tickers: {best_tickers[:10]}... (force_refresh={force_refresh})")
            save_to_cache(cache_key, best_tickers)
            return best_tickers
        print(f"[scrape_index_constituents] {index_name}: No valid table/column found, using core fallback list. (force_refresh={force_refresh})")
        core = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        save_to_cache(cache_key, core)
        return core
    except Exception as e:
        print(f"Error scraping {index_name}: {e}")
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

def download_index_data(index_name, start_date, force_refresh=False):
    # Always force refresh for debugging
    tickers = scrape_index_constituents(index_name, force_refresh=True)
    etf_ticker = INDEX_ETF_TICKERS.get(index_name)
    if etf_ticker and etf_ticker in tickers:
        tickers.remove(etf_ticker)
    if etf_ticker:
        tickers = [etf_ticker] + tickers
    print(f"[download_index_data] Batch downloading data for tickers: {tickers}")
    end_date = datetime.today()
    try:
        df = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
        stock_data = {}
        fallback_to_threadpool = False
        if df is None or df.empty or (len(tickers) > 1 and not isinstance(df.columns, pd.MultiIndex)):
            print("[download_index_data] Batch download failed or returned empty. Using ThreadPoolExecutor for per-ticker download.")
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def fetch_ticker(ticker):
                try:
                    tdf = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if tdf is not None and not tdf.empty and len(tdf) >= 50:
                        return ticker, tdf
                except Exception as e:
                    print(f"[download_index_data] Per-ticker download error for {ticker}: {e}")
                return ticker, None
            with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
                futures = {executor.submit(fetch_ticker, ticker): ticker for ticker in tickers}
                for future in as_completed(futures):
                    ticker, tdf = future.result()
                    if tdf is not None:
                        stock_data[ticker] = tdf
        else:
            for ticker in tickers:
                tdf = None
                if hasattr(df, 'columns') and ticker in df.columns.get_level_values(0):
                    try:
                        tdf = df[ticker].dropna()
                    except Exception:
                        tdf = None
                elif hasattr(df, 'columns') and ticker in df.columns.get_level_values(1):
                    try:
                        tdf = df.xs(ticker, axis=1, level=1, drop_level=False).dropna()
                    except Exception:
                        tdf = None
                if tdf is not None and not tdf.empty and len(tdf) >= 50:
                    stock_data[ticker] = tdf
        failed_downloads = [t for t in tickers if t not in stock_data]
        successful_downloads = len(stock_data)
        print(f"[download_index_data] Download complete. Success: {successful_downloads}, Failed: {failed_downloads}")
        return stock_data, fallback_to_threadpool, successful_downloads, failed_downloads
    except Exception as e:
        print(f"[download_index_data] Batch download error: {e}")
        return {}, False, 0, tickers

def calculate_market_condition(df):
    """Determine market condition based on price data"""
    if df is None or len(df) < 20:
        return 'sideways', 0.5
    
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Get recent data
    recent = df.tail(30)
    
    # Calculate trend strength
    price_trend = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
    sma_trend = (recent['SMA_20'].iloc[-1] - recent['SMA_20'].iloc[0]) / recent['SMA_20'].iloc[0]
    avg_volatility = recent['Volatility'].mean()
    
    # Determine market condition
    if abs(price_trend) < 0.05 and abs(sma_trend) < 0.03:
        condition = 'sideways'
        strength = 0.3
    elif price_trend > 0.1 and sma_trend > 0.05:
        condition = 'bull'
        strength = min(0.9, 0.5 + abs(price_trend))
    elif price_trend < -0.1 and sma_trend < -0.05:
        condition = 'bear'
        strength = min(0.9, 0.5 + abs(price_trend))
    elif avg_volatility > 0.03:
        condition = 'volatile'
        strength = min(0.8, 0.4 + avg_volatility * 10)
    else:
        condition = 'sideways'
        strength = 0.5
    
    return condition, strength

def select_models_for_market(market_condition, is_custom=False):
    """Select appropriate models based on market condition"""
    if is_custom:
        # For custom tickers, use a balanced selection
        return [2, 7, 6]  # Random Forest, SVR, Bayesian Ridge
    
    model_selections = {
        'bull': [1, 4, 8],      # XGBoost, Extra Trees, Gradient Boosting
        'bear': [6, 9, 2],      # Bayesian Ridge, Elastic Net, Random Forest
        'sideways': [2, 7, 6],  # Random Forest, SVR, Bayesian Ridge
        'volatile': [3, 5, 8]   # Neural Network, AdaBoost, Gradient Boosting
    }
    
    return model_selections.get(market_condition, [2, 7, 6])

def flatten_series(s):
    # If DataFrame, take first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    # If numpy array, flatten to 1D
    if isinstance(s, np.ndarray):
        s = s.flatten()
        return pd.Series(s)
    # If Series, ensure it's 1D
    if isinstance(s, pd.Series):
        return s
    # Fallback: convert to 1D array then Series
    arr = np.asarray(s).flatten()
    return pd.Series(arr)

def detect_market_regime(etf_df):
    """Robust regime detection using multiple indicators from the index ETF ticker's data."""
    if etf_df is None or len(etf_df) < 50:
        return 'sideways', 0.5
    df = etf_df.copy()
    # Calculate indicators if not present
    if 'SMA_20' not in df:
        df['SMA_20'] = flatten_series(df['Close'].rolling(window=20).mean())
    if 'SMA_50' not in df:
        df['SMA_50'] = flatten_series(df['Close'].rolling(window=50).mean())
    if 'macd' not in df or 'macd_signal' not in df:
        from ta.trend import MACD
        macd = MACD(close=df['Close'])
        macd_val_raw = macd.macd()
        macd_signal_raw = macd.macd_signal()
        df['macd'] = flatten_series(macd_val_raw)
        df['macd_signal'] = flatten_series(macd_signal_raw)
        df['ema_diff'] = flatten_series(df['macd'] - df['macd_signal'])
    if 'stoch_k' not in df:
        from ta.momentum import StochasticOscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = flatten_series(stoch.stoch())
    if 'Donchian_Width' not in df:
        df['Donchian_Width'] = flatten_series(df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())
    if 'RSI' not in df:
        df['RSI'] = flatten_series(RSIIndicator(close=df['Close']).rsi())
    if 'williams_r' not in df:
        from ta.momentum import WilliamsRIndicator
        df['williams_r'] = flatten_series(WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r())
    if 'ATR' not in df:
        df['ATR'] = flatten_series(AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range())
    if 'rolling_20d_std' not in df:
        df['rolling_20d_std'] = flatten_series(df['Close'].rolling(window=20).std())
    if 'percent_b' not in df:
        upper = flatten_series(df['SMA_20'] + 2*df['Close'].rolling(window=20).std())
        lower = flatten_series(df['SMA_20'] - 2*df['Close'].rolling(window=20).std())
        df['percent_b'] = flatten_series((df['Close'] - lower) / (upper - lower))
    if 'OBV' not in df:
        df['OBV'] = flatten_series(OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume())
    if 'cmf' not in df:
        from ta.volume import ChaikinMoneyFlowIndicator
        df['cmf'] = flatten_series(ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow())
    recent = df.tail(30)
    # --- Trend ---
    sma_20 = recent['SMA_20'].iloc[-1]
    sma_50 = recent['SMA_50'].iloc[-1]
    price = recent['Close'].iloc[-1]
    macd_val = recent['macd'].iloc[-1]
    macd_signal = recent['macd_signal'].iloc[-1]
    ema_diff = recent['ema_diff'].iloc[-1]
    donchian_width = recent['Donchian_Width'].iloc[-1]
    # --- Momentum ---
    rsi = recent['RSI'].iloc[-1]
    willr = recent['williams_r'].iloc[-1]
    stoch_k = recent['stoch_k'].iloc[-1]
    # --- Volatility ---
    atr = recent['ATR'].iloc[-1]
    rolling_std = recent['rolling_20d_std'].iloc[-1]
    percent_b = recent['percent_b'].iloc[-1]
    # --- Volume ---
    obv = recent['OBV'].iloc[-1]
    cmf = recent['cmf'].iloc[-1]
    # --- Regime logic ---
    # Trend regime (use macd and ema_diff instead of price_vs_ema)
    bull = (price > sma_20 > sma_50) and (macd_val > macd_signal) and (ema_diff > 0) and (donchian_width > 0.02 * price)
    bear = (price < sma_20 < sma_50) and (macd_val < macd_signal) and (ema_diff < 0) and (donchian_width > 0.02 * price)
    # Momentum regime
    overbought = (rsi > 70) or (willr > -20) or (stoch_k > 80)
    oversold = (rsi < 30) or (willr < -80) or (stoch_k < 20)
    # Volatility regime
    high_vol = (atr > 0.03 * price) or (rolling_std > 0.03 * price) or (percent_b > 0.95 or percent_b < 0.05)
    # Volume regime
    strong_volume = (cmf > 0.1) or (obv > 0)
    weak_volume = (cmf < -0.1) or (obv < 0)
    # Decision
    if bull and not high_vol and strong_volume:
        return 'bull', 0.8
    elif bear and not high_vol and weak_volume:
        return 'bear', 0.8
    elif high_vol:
        return 'volatile', 0.7
    elif overbought or oversold:
        return 'sideways', 0.5
    else:
        return 'sideways', 0.5

def calc_slope(x):
    """Calculate slope of a series"""
    if len(x) < 2:
        return np.nan
    return np.polyfit(range(len(x)), x, 1)[0]

def add_features_to_stock(ticker, df, prediction_window=5):
    """Add features to a single stock's data, with N-day lookahead and 30 total indicators (5 per group)."""
    if df is None or len(df) < 50:
        return None
    try:
        df = df.copy()
        df['close_series'] = df['Close']
        df['high_series'] = df['High']
        df['low_series'] = df['Low']
        df['volume_series'] = df['Volume']
        # --- Momentum ---
        df['RSI'] = flatten_series(RSIIndicator(close=df['close_series']).rsi())
        df['roc_10'] = flatten_series(ROCIndicator(close=df['close_series']).roc())
        df['momentum'] = flatten_series(df['close_series'] - df['close_series'].shift(10))
        df['rsi_14_diff'] = flatten_series(df['RSI'] - df['RSI'].shift(5))
        from ta.momentum import WilliamsRIndicator
        df['williams_r'] = flatten_series(WilliamsRIndicator(high=df['high_series'], low=df['low_series'], close=df['close_series'], lbp=14).williams_r())
        # --- Mean Reversion ---
        df['SMA_20'] = flatten_series(df['close_series'].rolling(window=20).mean().shift(1))
        df['SD'] = flatten_series(df['close_series'].rolling(window=20).std().shift(1))
        df['Upper'] = flatten_series(df['SMA_20'] + 2*df['SD'])
        df['Lower'] = flatten_series(df['SMA_20'] - 2*df['SD'])
        df['percent_b'] = flatten_series((df['close_series'] - df['Lower']) / (df['Upper'] - df['Lower']))
        rolling_mean = flatten_series(df['close_series'].rolling(window=10).mean())
        df['z_score_close'] = flatten_series((df['close_series'] - rolling_mean) / df['close_series'].rolling(window=10).std())
        # --- Trend ---
        from ta.trend import MACD, EMAIndicator
        macd = MACD(close=df['close_series'])
        df['macd'] = flatten_series(macd.macd().squeeze())
        df['macd_signal'] = flatten_series(macd.macd_signal().squeeze())
        df['ema_diff'] = flatten_series((df['macd'] - df['macd_signal']).squeeze())
        # Calculate slope_price_10d: slope of close price over last 10 days
        def calc_slope(x):
            if len(x) < 2 or np.any(np.isnan(x)):
                return np.nan
            y = np.array(x)
            x_vals = np.arange(len(y))
            A = np.vstack([x_vals, np.ones(len(x_vals))]).T
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return m
        df['slope_price_10d'] = flatten_series(df['close_series'].rolling(window=10).apply(calc_slope, raw=False))
        # Calculate ema_ratio: ratio of close to EMA_10
        ema_10 = flatten_series(EMAIndicator(close=df['close_series'], window=10).ema_indicator())
        df['ema_ratio'] = flatten_series(df['close_series'] / ema_10)
        # Normalize MACD features to mean=0, std=1 (z-score)
        for col in ['macd', 'macd_signal', 'ema_diff']:
            series = df[col]
            df[col] = flatten_series((series - series.mean()) / (series.std() if series.std() != 0 else 1))
        # --- Momentum ---
        from ta.momentum import StochasticOscillator
        stoch = StochasticOscillator(high=df['high_series'], low=df['low_series'], close=df['close_series'])
        df['stoch_k'] = flatten_series(stoch.stoch().squeeze())
        # Normalize stoch_k to 0-100
        if df['stoch_k'].max() != df['stoch_k'].min():
            df['stoch_k'] = flatten_series(100 * (df['stoch_k'] - df['stoch_k'].min()) / (df['stoch_k'].max() - df['stoch_k'].min()))
        else:
            df['stoch_k'] = flatten_series(50)  # fallback if constant
        # --- Volatility ---
        from ta.volatility import BollingerBands
        df['ATR'] = flatten_series(AverageTrueRange(high=df['high_series'], low=df['low_series'], close=df['close_series']).average_true_range())
        df['volatility'] = flatten_series(np.log(df['close_series'] / df['close_series'].shift(1)).rolling(10).std())
        df['rolling_20d_std'] = flatten_series(df['close_series'].rolling(window=20).std())
        bb = BollingerBands(close=df['close_series'])
        df['bb_width'] = flatten_series(bb.bollinger_wband())
        df['donchian_width'] = flatten_series(df['high_series'].rolling(window=20).max() - df['low_series'].rolling(window=20).min())
        # --- Volume ---
        df['OBV'] = flatten_series(OnBalanceVolumeIndicator(close=df['close_series'], volume=df['volume_series']).on_balance_volume())
        df['volume_pct_change'] = flatten_series(df['Volume'].pct_change())
        from ta.volume import ChaikinMoneyFlowIndicator, AccDistIndexIndicator
        df['cmf'] = flatten_series(ChaikinMoneyFlowIndicator(high=df['high_series'], low=df['low_series'], close=df['close_series'], volume=df['volume_series']).chaikin_money_flow())
        df['adl'] = flatten_series(AccDistIndexIndicator(high=df['high_series'], low=df['low_series'], close=df['close_series'], volume=df['volume_series']).acc_dist_index())
        # --- Other/Composite ---
        df['close_lag_1'] = flatten_series(df['Close'].shift(1))
        df['close_lag_5'] = flatten_series(df['Close'].shift(5))
        df['past_10d_return'] = flatten_series(df['close_series'] / df['close_series'].shift(10) - 1)
        df[f'close_lead_{prediction_window}'] = flatten_series(df['Close'].shift(-prediction_window))
        df[f'forward_return_{prediction_window}'] = flatten_series((df['Close'].shift(-prediction_window) / df['Close']) - 1)
        df[f'rolling_max_{prediction_window}'] = flatten_series(df['Close'].rolling(window=prediction_window).max())
        df[f'rolling_min_{prediction_window}'] = flatten_series(df['Close'].rolling(window=prediction_window).min())
        # --- Ichimoku ---
        from ta.trend import IchimokuIndicator
        ichimoku = IchimokuIndicator(high=df['high_series'], low=df['low_series'], window1=9, window2=26, window3=52, fillna=True)
        df['ichimoku_a'] = flatten_series(ichimoku.ichimoku_a())
        df['ichimoku_b'] = flatten_series(ichimoku.ichimoku_b())
        df['ichimoku_base'] = flatten_series(ichimoku.ichimoku_base_line())
        # Drop rows with NaN for rolling features, but NOT for close_lead_N, forward_return_N, rolling_max/min
        feature_cols = [
            'RSI', 'roc_10', 'momentum', 'rsi_14_diff', 'williams_r', 'stoch_k',
            'SMA_20', 'SD', 'Upper', 'Lower', 'percent_b', 'z_score_close',
            'macd', 'macd_signal', 'slope_price_10d', 'ema_diff', 'ema_ratio',
            'ATR', 'volatility', 'rolling_20d_std', 'bb_width', 'donchian_width',
            'OBV', 'volume_pct_change', 'cmf', 'adl', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base',
            'close_lag_1', 'close_lag_5', 'past_10d_return'
        ]
        df = df.dropna(subset=feature_cols)
        return df
    except Exception as e:
        return None

def add_features_parallel(stock_data, prediction_window=5):
    """Add features to all stocks using parallel processing, with N-day lookahead."""
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_ticker = {
            executor.submit(add_features_to_stock, ticker, df, prediction_window): ticker 
            for ticker, df in stock_data.items()
        }
        processed_data = {}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                processed_df = future.result()
                if processed_df is not None and len(processed_df) > 20:
                    processed_data[ticker] = processed_df
            except Exception as e:
                print(f"Error processing features for {ticker}: {e}")
    return processed_data

def get_indicator_weights(regime, regime_strength):
    """Return a dict of feature weights based on regime and regime strength."""
    # Define base weights for each group per regime
    base_weights = {
        'bull': {'momentum': 1.3, 'trend': 1.2, 'mean_reversion': 0.8, 'volatility': 0.7, 'volume': 1.0, 'other': 1.0},
        'bear': {'momentum': 0.8, 'trend': 0.7, 'mean_reversion': 1.3, 'volatility': 1.2, 'volume': 1.0, 'other': 1.0},
        'sideways': {'momentum': 0.8, 'trend': 0.8, 'mean_reversion': 1.3, 'volatility': 1.0, 'volume': 1.0, 'other': 1.0},
        'volatile': {'momentum': 0.7, 'trend': 0.7, 'mean_reversion': 1.1, 'volatility': 1.3, 'volume': 1.2, 'other': 1.0},
    }
    # Map each feature to a group
    group_map = {
        # Momentum
        'RSI': 'momentum', 'roc_10': 'momentum', 'momentum': 'momentum', 'rsi_14_diff': 'momentum', 'williams_r': 'momentum', 'stoch_k': 'momentum',
        # Mean Reversion
        'SMA_20': 'mean_reversion', 'SD': 'mean_reversion', 'Upper': 'mean_reversion', 'Lower': 'mean_reversion', 'percent_b': 'mean_reversion', 'z_score_close': 'mean_reversion',
        # Trend
        'macd': 'trend', 'macd_signal': 'trend', 'slope_price_10d': 'trend', 'ema_diff': 'trend', 'ema_ratio': 'trend',
        # Volatility
        'ATR': 'volatility', 'volatility': 'volatility', 'rolling_20d_std': 'volatility', 'bb_width': 'volatility', 'donchian_width': 'volatility',
        # Volume
        'OBV': 'volume', 'volume_pct_change': 'volume', 'cmf': 'volume', 'adl': 'volume', 'ichimoku_a': 'other', 'ichimoku_b': 'other', 'ichimoku_base': 'other',
        # Other/Composite
        'close_lag_1': 'other', 'close_lag_5': 'other', 'past_10d_return': 'other',
    }
    regime = regime if regime in base_weights else 'sideways'
    weights = {}
    for feature, group in group_map.items():
        regime_weight = base_weights[regime][group]
        final_weight = 1 + (regime_weight - 1) * (regime_strength - 0.5) * 2
        weights[feature] = min(1.4, max(0.7, final_weight))
    return weights

def train_model_for_stock(ticker, df, model_ids, regime=None, regime_strength=0.5):
    """Train models for a single stock, with regime-aware feature weighting."""
    if df is None or len(df) < 50:
        return None
    try:
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        # Forcibly flatten all feature columns to 1D, then handle object dtype
        for col in feature_columns:
            df[col] = np.asarray(df[col]).reshape(-1)
            if df[col].dtype == 'O':
                df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) == 1 else (x if np.isscalar(x) else float(np.asarray(x).flatten()[0])))
            df[col] = pd.to_numeric(df[col], errors='coerce')
        X = df[feature_columns].values.astype(float)
        y = df['Close'].pct_change().shift(-1).values  # Next day return
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        if len(X) < 20:
            return None
        if regime is not None:
            weights_dict = get_indicator_weights(regime, regime_strength)
            weights = [weights_dict.get(f, 1.0) for f in feature_columns]
            X = X * weights
        model_predictions = {}
        for model_id in model_ids:
            prediction = train_and_predict_model(X, y, model_id)
            model_predictions[model_id] = prediction
        return model_predictions
    except Exception as e:
        print(f"Error training models for {ticker}: {e}")
        return None

class TransformerModel:
    """Transformer-like model using scikit-learn components"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2):
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        # Use MLPRegressor as transformer alternative
        self.model = MLPRegressor(
            hidden_layer_sizes=(d_model, d_model//2, d_model//4),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def train_and_predict_model(X, y, model_id):
    """Train actual ML model and make prediction, with feature scaling."""
    if len(X) < 50:  # Need sufficient data
        return 0.0
    try:
        # Prepare data
        X = np.array(X)
        y = np.array(y)
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        if len(X) < 20:
            return 0.0
        # Split data (use all historical data for training, predict current)
        X_train = X[:-1]  # All but last row
        y_train = y[:-1]  # All but last row
        X_predict = X[-1:]  # Last row for prediction
        # Choose scaler
        if model_id in [3, 6, 7, 9, 10]:  # Neural net, regression, SVR, ElasticNet, Transformer
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)
        # Train model based on model_id
        if model_id == 1:  # XGBoost Quantile Regression
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=0.5,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 2:  # Random Forest Bootstrap
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                bootstrap=True,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 3:  # Neural Network Conformal
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 4:  # Extra Trees Bootstrap
            model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                bootstrap=True,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 5:  # AdaBoost Conformal
            model = AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 6:  # Bayesian Ridge Conformal
            model = BayesianRidge(
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 7:  # Support Vector Regression
            model = SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 8:  # Gradient Boosting Conformal
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 9:  # Elastic Net Conformal
            model = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        elif model_id == 10:  # Transformer (scikit-learn alternative)
            model = TransformerModel(input_dim=X_train_scaled.shape[1])
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(X_predict_scaled)[0]
        else:
            return 0.0
        return prediction
    except Exception as e:
        print(f"Error training model {model_id}: {e}")
        return 0.0

def train_models_parallel(stock_data, model_ids, regime=None, regime_strength=0.5):
    """Train models for all stocks using parallel processing, with regime-aware feature weighting."""
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_ticker = {
            executor.submit(train_model_for_stock, ticker, df, model_ids, regime, regime_strength): ticker 
            for ticker, df in stock_data.items()
        }
        
        trained_models = {}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                model_predictions = future.result()
                if model_predictions is not None:
                    trained_models[ticker] = model_predictions
            except Exception as e:
                print(f"Error training models for {ticker}: {e}")
    
    return trained_models

def simulate_prediction_model(df, model_id, prediction_window, confidence_interval):
    """Simulate prediction using different model types"""
    if df is None or len(df) < 50:
        return None, None, None
    
    # Calculate all technical indicators
    df = add_features_to_stock('TICKER', df, prediction_window)
    
    # Get recent data for prediction
    recent = df.tail(30).dropna()
    
    if len(recent) < 20:
        return None, None, None
    
    # Simulate different model behaviors based on technical indicators
    base_prediction = np.random.normal(0.02, 0.05)  # 2% average return
    
    # Use technical indicators to adjust prediction
    if not recent.empty:
        # RSI-based adjustment
        current_rsi = recent['RSI'].iloc[-1] if not pd.isna(recent['RSI'].iloc[-1]) else 50
        if current_rsi > 70:
            base_prediction *= 0.8  # Overbought, reduce prediction
        elif current_rsi < 30:
            base_prediction *= 1.2  # Oversold, increase prediction
        
        # Momentum-based adjustment
        momentum = recent['momentum'].iloc[-1] if not pd.isna(recent['momentum'].iloc[-1]) else 0
        if momentum > 0:
            base_prediction *= 1.1
        else:
            base_prediction *= 0.9
        
        # Volatility-based adjustment
        volatility = recent['volatility'].iloc[-1] if not pd.isna(recent['volatility'].iloc[-1]) else 0.02
        if volatility > 0.03:
            base_prediction *= 0.8  # High volatility, reduce prediction
        elif volatility < 0.01:
            base_prediction *= 1.1  # Low volatility, increase prediction
    
    # Adjust based on model type
    if model_id in [1, 4, 8]:  # Aggressive models
        base_prediction *= 1.5
    elif model_id in [6, 9]:   # Conservative models
        base_prediction *= 0.7
    
    # Add market condition influence
    market_condition, _ = calculate_market_condition(df)
    if market_condition == 'bull':
        base_prediction += 0.01
    elif market_condition == 'bear':
        base_prediction -= 0.01
    
    # Calculate confidence interval
    confidence_factor = confidence_interval / 100.0
    interval_width = 0.05 * (1 - confidence_factor)  # Wider interval for lower confidence
    
    lower = base_prediction - interval_width
    upper = base_prediction + interval_width
    
    return base_prediction, lower, upper

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator (fallback method)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_stock_predictions(index_ticker, num_stocks, start_date, prediction_window, confidence_interval):
    """Generate predictions for multiple stocks in an index"""
    # Get index components (simplified - in real app you'd get actual components)
    if index_ticker == 'SPY':
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM']
    elif index_ticker == 'DOW':
        tickers = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'UNH', 'HD', 'DIS']
    elif index_ticker == 'NASDAQ':
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM']
    else:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    predictions = []
    
    for ticker in tickers[:num_stocks]:
        df = download_index_data(ticker, start_date)
        if df is not None and len(df) > 50:
            pred, lower, upper = simulate_prediction_model(df, 2, prediction_window, confidence_interval)
            if pred is not None:
                predictions.append({
                    'ticker': ticker,
                    'pred': pred,
                    'lower': lower,
                    'upper': upper
                })
    
    # Sort by prediction value
    predictions.sort(key=lambda x: x['pred'], reverse=True)
    return predictions

def create_prediction_chart(df, prediction, lower, upper, ticker_name):
    """Create a matplotlib chart showing prediction"""
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # Plot historical data
    if df is not None and len(df) > 0:
        dates = df.index[-50:]  # Last 50 days
        prices = df['Close'][-50:]
        plt.plot(dates, prices, 'white', linewidth=2, label='Historical Price')
    
    # Add prediction point
    if df is not None and len(df) > 0:
        last_date = df.index[-1]
        future_date = last_date + timedelta(days=5)
        current_price = df['Close'].iloc[-1]
        
        # Plot prediction
        plt.scatter(future_date, current_price * (1 + prediction), 
                   color='green', s=100, zorder=5, label=f'Prediction: {prediction*100:.2f}%')
        
        # Plot confidence interval
        plt.fill_between([future_date], 
                        current_price * (1 + lower), 
                        current_price * (1 + upper), 
                        alpha=0.3, color='green', label=f'Confidence Interval')
    
    plt.title(f'{ticker_name} - 5-Day Price Prediction', color='white', fontsize=16)
    plt.xlabel('Date', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                facecolor='#1e293b', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def create_multi_stock_prediction_chart(stock_data, stock_predictions, prediction_window):
    """Plot all top N stocks: last X days of history and X days of forecast, normalized to 100 at -X. Fix gap between history and prediction."""
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    colors = cm.get_cmap('tab10', len(stock_predictions))
    for idx, stock in enumerate(stock_predictions):
        ticker = stock['ticker']
        if ticker not in stock_data:
            continue
        df = stock_data[ticker]
        if len(df) < prediction_window + 1:
            continue
        # Get last X days of history
        hist = df['Close'].iloc[-prediction_window-1:]
        # Normalize to 100 at -X
        norm_factor = hist.iloc[0]
        hist_norm = hist / norm_factor * 100
        # Simulate predicted path (start at last hist value)
        pred_path = [hist_norm.iloc[-1]]
        for i in range(1, prediction_window+1):
            pred_path.append(pred_path[-1] * (1 + stock['pred']))
        # x-axis: -X to 0 for history, 0 to X for forecast
        x_hist = np.arange(-prediction_window, 1)
        x_pred = np.arange(0, prediction_window+1)
        # Only label the solid line for the legend
        plt.plot(x_hist, hist_norm.values, color=colors(idx), linewidth=2, label=f"{ticker}")
        plt.plot(x_pred, pred_path, color=colors(idx), linewidth=2, linestyle='--')
    # Add vertical dashed white line at x=0
    plt.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel(f"Days (0 = present, -N = history, N = forecast)", color='white')
    plt.ylabel("Normalized Price (Start = 100)", color='white')
    plt.title(f"Top {len(stock_predictions)} Stock Predictions (window={prediction_window} days)", color='white', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

# --- Single Ticker Functions (no parallelization) ---
def download_single_ticker_data(ticker, start_date):
    """Download data for a single ticker."""
    import yfinance as yf
    end_date = datetime.today()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Flatten MultiIndex columns if present and only one ticker
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            if len(df.columns.levels[1]) == 1:
                df.columns = df.columns.droplevel(1)
        if df is not None and not df.empty and len(df) >= 50:
            return df
        else:
            return None
    except Exception as e:
        print(f"[download_single_ticker_data] Error downloading {ticker}: {e}")
        return None

def to_scalar(x):
    if isinstance(x, (np.ndarray, list)):
        x = np.asarray(x).flatten()
        if x.size == 0:
            return np.nan
        return float(x[0])
    elif pd.isna(x):
        return np.nan
    else:
        return float(x)

def add_features_single(ticker, df, prediction_window=5):
    """Add features to a single ticker's data (no parallelization)."""
    return add_features_to_stock(ticker, df, prediction_window)

def train_model_single(df, model_ids, regime=None, regime_strength=0.5):
    """Train models for a single ticker (no parallelization)."""
    if df is None or len(df) < 50:
        return None
    try:
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        flat_features = {}
        for col in feature_columns:
            arr = df[col]
            arr_flat = [to_scalar(x) for x in arr]
            arr_flat = pd.to_numeric(arr_flat, errors='coerce')
            flat_features[col] = arr_flat
        features_df = pd.DataFrame(flat_features, index=df.index)
        X = features_df[feature_columns].values.astype(float)
        y = df['Close'].pct_change().shift(-1).values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        if len(X) < 20:
            return None
        if regime is not None:
            weights_dict = get_indicator_weights(regime, regime_strength)
            weights = [weights_dict.get(f, 1.0) for f in feature_columns]
            X = X * weights
        model_predictions = {}
        for model_id in model_ids:
            prediction = train_and_predict_model(X, y, model_id)
            model_predictions[model_id] = prediction
        return model_predictions
    except Exception as e:
        return None

def predict_single_ticker_chart(df, predictions, prediction_window):
    """Create a chart for a single ticker: last X days of history and X days of forecast, normalized to 100 at -X."""
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    if df is None or len(df) < prediction_window + 1:
        return None
    hist = df['Close'].iloc[-prediction_window-1:]
    norm_factor = hist.iloc[0]
    hist_norm = hist / norm_factor * 100
    avg_pred = np.mean(list(predictions.values()))
    # History
    x_hist = np.arange(-prediction_window, 1)
    plt.plot(x_hist, hist_norm.values, color='cyan', linewidth=2, label="History")
    # Prediction (dashed, same color)
    pred_start = hist_norm.values[-1]
    pred_path = [pred_start]
    for i in range(1, prediction_window+1):
        pred_path.append(pred_path[-1] * (1 + avg_pred))
    x_pred = np.arange(0, prediction_window+1)
    plt.plot(x_pred, pred_path, color='cyan', linewidth=2, linestyle='--', label="Prediction")
    plt.xlabel(f"Days (0 = present, -N = history, N = forecast)", color='white')
    plt.ylabel("Normalized Price (Start = 100)", color='white')
    plt.title(f"Single Ticker Prediction (window={prediction_window} days)", color='white', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='#1e293b', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        force_refresh = False
        if request.args.get('force_refresh', '').lower() == 'true':
            force_refresh = True
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        # Extract parameters
        index = data.get('index', 'SPY')
        num_stocks = data.get('numStocks', 10)
        custom_ticker = data.get('customTicker', '')
        start_date = data.get('startDate', '2024-01-01')
        prediction_window = data.get('predictionWindow', 5)
        confidence_interval = data.get('confidenceInterval', 70)
        model_selection = data.get('modelSelection', 'auto')
        selected_models = data.get('selectedModels', [])
        is_custom = index == 'CUSTOM'
        if is_custom:
            ticker_to_analyze = custom_ticker
            if not ticker_to_analyze:
                return jsonify({'error': 'Please enter a valid ticker symbol (e.g., AAPL, MSFT)'}), 400
            spy_df = download_single_ticker_data('SPY', start_date)
            if spy_df is not None:
                market_condition, market_strength = detect_market_regime(spy_df)
            else:
                market_condition, market_strength = 'sideways', 0.5
            if model_selection == 'auto':
                selected_models = select_models_for_market(market_condition, is_custom)
            elif not selected_models:
                selected_models = [2, 7, 6]
            df = download_single_ticker_data(ticker_to_analyze, start_date)
            if df is None or df.empty or len(df) < 50:
                return jsonify({'error': f'❌ Could not fetch data for {ticker_to_analyze}. Please check the ticker symbol and try again.'}), 400
            features_df = add_features_single(ticker_to_analyze, df, prediction_window)
            if features_df is None or features_df.empty:
                return jsonify({'error': '❌ Could not process technical indicators. Data may be corrupted or insufficient.'}), 400
            model_preds = train_model_single(features_df, selected_models, market_condition, market_strength)
            if not model_preds:
                return jsonify({'error': '❌ Could not train machine learning models. Insufficient data or technical issues.'}), 400
            avg_pred = np.mean(list(model_preds.values()))
            confidence_factor = confidence_interval / 100.0
            interval_width = 0.05 * (1 - confidence_factor)
            lower = avg_pred - interval_width
            upper = avg_pred + interval_width
            chart_image = predict_single_ticker_chart(features_df, model_preds, prediction_window)
            response = {
                'index_prediction': {
                    'ticker': ticker_to_analyze,
                    'index_name': ticker_to_analyze,
                    'pred': avg_pred,
                    'lower': lower,
                    'upper': upper
                },
                'selected_models': selected_models,
                'market_condition': market_condition,
                'market_strength': market_strength,
                'plot_image': chart_image,
                'system_messages': []
            }
            return jsonify(response)
        else:
            print(f"Starting multi-ticker analysis for {index} with {num_stocks} stocks...")
            print("Downloading stock data...")
            result = download_index_data(index, start_date, force_refresh=force_refresh)
            if isinstance(result, tuple):
                stock_data, fallback_used, successful_downloads, failed_downloads = result
            else:
                stock_data = result
                fallback_used = False
                successful_downloads = len(stock_data) if stock_data else 0
                failed_downloads = []
            if successful_downloads == 0:
                error_msg = f'Could not download any data for {index}. Network or data source issues.'
                print(f"ERROR: {error_msg}")
                return jsonify({'error': error_msg}), 400
            print(f"Downloaded data for {len(stock_data)} stocks (skipped {len(failed_downloads)} failed tickers)")
            print("Adding technical features...")
            processed_data = add_features_parallel(stock_data, prediction_window)
            if not processed_data:
                print(f"ERROR: Could not process features")
                return jsonify({'error': '❌ Could not process technical indicators. Data may be corrupted or insufficient.'}), 400
            print(f"Added features to {len(processed_data)} stocks")
            etf_ticker = INDEX_ETF_TICKERS.get(index)
            etf_df = stock_data.get(etf_ticker)
            if etf_df is not None:
                market_condition, market_strength = detect_market_regime(etf_df)
            else:
                first_stock = list(processed_data.values())[0]
                market_condition, market_strength = detect_market_regime(first_stock)
            if model_selection == 'auto':
                selected_models = select_models_for_market(market_condition, is_custom)
            elif not selected_models:
                selected_models = [2, 7, 6]
            print("Training models and generating predictions...")
            trained_models = train_models_parallel(processed_data, selected_models, market_condition, market_strength)
            if not trained_models:
                print(f"ERROR: Could not train models")
                return jsonify({'error': '❌ Could not train machine learning models. Insufficient data or technical issues.'}), 400
            print(f"Trained models for {len(trained_models)} stocks")
            stock_predictions = []
            for ticker, model_predictions in trained_models.items():
                if etf_ticker and ticker == etf_ticker:
                    continue  # Skip the ETF ticker in stock_predictions
                avg_prediction = np.mean(list(model_predictions.values()))
                confidence_factor = confidence_interval / 100.0
                interval_width = 0.05 * (1 - confidence_factor)
                lower = avg_prediction - interval_width
                upper = avg_prediction + interval_width
                stock_predictions.append({
                    'ticker': ticker,
                    'pred': avg_prediction,
                    'lower': lower,
                    'upper': upper
                })
            stock_predictions.sort(key=lambda x: x['pred'], reverse=True)
            stock_predictions = stock_predictions[:num_stocks]
            
            # Use the ETF's own prediction for the main index prediction
            etf_prediction_result = trained_models.get(etf_ticker)
            if etf_ticker and etf_prediction_result:
                index_prediction = np.mean(list(etf_prediction_result.values()))
                confidence_factor = confidence_interval / 100.0
                interval_width = 0.05 * (1 - confidence_factor)
                index_lower = index_prediction - interval_width
                index_upper = index_prediction + interval_width
                index_name_for_response = etf_ticker
            else:
                # Fallback to averaging top 5 stocks if ETF prediction is not available
                index_prediction = np.mean([s['pred'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_lower = np.mean([s['lower'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_upper = np.mean([s['upper'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_name_for_response = index

            chart_image = create_multi_stock_prediction_chart(stock_data, stock_predictions, prediction_window)
            response = {
                'index_prediction': {
                    'ticker': index_name_for_response,
                    'index_name': index_name_for_response,
                    'pred': index_prediction,
                    'lower': index_lower,
                    'upper': index_upper
                },
                'selected_models': selected_models,
                'market_condition': market_condition,
                'market_strength': market_strength,
                'plot_image': chart_image,
                'stock_predictions': stock_predictions,
                'system_messages': []
            }
            if fallback_used:
                response['system_messages'].append({
                    'type': 'warning',
                    'message': f'⚠️ Using fallback ticker list for {index} (Wikipedia scraping unavailable)'
                })
            if failed_downloads:
                response['system_messages'].append({
                    'type': 'info',
                    'message': f'📊 Successfully downloaded {successful_downloads} stocks. Skipped {len(failed_downloads)} failed tickers: {", ".join(failed_downloads[:3])}'
                })
            if len(stock_data) < num_stocks:
                response['system_messages'].append({
                    'type': 'info',
                    'message': f'📈 Analyzed {len(stock_data)} stocks (requested {num_stocks})'
                })
        return jsonify(response)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'IndexLab Backend is running'})

if __name__ == '__main__':
    print("Starting IndexLab Backend Server...")
    print("Available endpoints:")
    print("- POST /api/predict - Main prediction endpoint")
    print("- GET /api/health - Health check")
    app.run(debug=True, host='0.0.0.0', port=5000) 