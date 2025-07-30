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
import traceback
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
warnings.filterwarnings('ignore')
from ta.trend import EMAIndicator
from ta.trend import IchimokuIndicator
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
# Removed TensorFlow and PyTorch imports - using scikit-learn alternatives
import matplotlib.cm as cm
from ta.momentum import WilliamsRIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, AccDistIndexIndicator
from bs4 import BeautifulSoup
from bs4.element import Tag
from scipy import stats
app = Flask(__name__)
CORS(app)

def ensure_iterable(obj):
    """Ensure the object is iterable. If not, wrap it in a list."""
    if isinstance(obj, (list, np.ndarray, pd.Series)):
        return obj
    return [obj]

def validate_prediction_bounds(prediction, ticker, historical_data=None, strict_mode=True):
    """
    Validate prediction against historical bounds and market reality.
    Returns validated prediction and warning flags.
    """
    try:
        original_pred = float(prediction)
        validated_pred = original_pred
        warnings = []
        
        # Define absolute bounds based on market reality
        if strict_mode:
            max_daily_return = 0.20   # 20% max daily move (very conservative)
            normal_daily_limit = 0.10  # 10% for normal market conditions
        else:
            max_daily_return = 0.30   # 30% max daily move (less conservative)
            normal_daily_limit = 0.15  # 15% for normal market conditions
        
        # Check absolute bounds
        if abs(original_pred) > max_daily_return:
            validated_pred = np.sign(original_pred) * max_daily_return
            warnings.append(f"EXTREME: Prediction {original_pred*100:.2f}% clamped to {validated_pred*100:.2f}%")
        elif abs(original_pred) > normal_daily_limit:
            # Apply soft clamping for large but not extreme predictions
            excess = abs(original_pred) - normal_daily_limit
            damping_factor = 0.5  # Reduce excess by 50%
            validated_pred = np.sign(original_pred) * (normal_daily_limit + excess * damping_factor)
            warnings.append(f"LARGE: Prediction {original_pred*100:.2f}% damped to {validated_pred*100:.2f}%")
        
        # Historical bounds checking if data is available
        if historical_data is not None and len(historical_data) > 50:
            try:
                # Calculate historical daily returns
                daily_returns = historical_data['Close'].pct_change().dropna()
                
                if len(daily_returns) > 30:
                    # Historical percentiles
                    hist_99 = np.percentile(daily_returns, 99)
                    hist_1 = np.percentile(daily_returns, 1)
                    hist_95 = np.percentile(daily_returns, 95)
                    hist_5 = np.percentile(daily_returns, 5)
                    
                    # Check against historical extremes (99th/1st percentile)
                    if validated_pred > hist_99 * 1.5:  # 1.5x historical extreme
                        validated_pred = hist_99 * 1.2  # Cap at 1.2x historical extreme
                        warnings.append(f"HISTORICAL: Prediction above 1.5x historical 99th percentile, capped")
                    elif validated_pred < hist_1 * 1.5:
                        validated_pred = hist_1 * 1.2
                        warnings.append(f"HISTORICAL: Prediction below 1.5x historical 1st percentile, capped")
                    
                    # Check against normal historical range (95th/5th percentile)
                    elif validated_pred > hist_95 * 2.0:
                        validated_pred = hist_95 * 1.5
                        warnings.append(f"RANGE: Prediction above 2x historical 95th percentile, reduced")
                    elif validated_pred < hist_5 * 2.0:
                        validated_pred = hist_5 * 1.5
                        warnings.append(f"RANGE: Prediction below 2x historical 5th percentile, reduced")
                        
            except Exception as e:
                warnings.append(f"Historical validation failed: {e}")
        
        # Sanity check: predictions should not be exactly zero unless intended
        if abs(validated_pred) < 1e-6 and abs(original_pred) > 1e-4:
            validated_pred = np.sign(original_pred) * 0.001  # Minimum 0.1% prediction
            warnings.append("ZERO: Tiny prediction increased to minimum threshold")
        
        # Log validation results if there were changes
        if warnings:
            print(f"[validate_prediction_bounds] {ticker}: {'; '.join(warnings)}")
        
        return float(validated_pred), warnings
        
    except Exception as e:
        print(f"[validate_prediction_bounds] Error for {ticker}: {e}")
        return float(prediction) if prediction is not None else 0.0, [f"Validation error: {e}"]

def sanitize_for_json(obj):
    """Recursively replace NaN and inf values with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    else:
        return obj

# --- Improved Confidence Interval Calculations ---
def calculate_proper_confidence_interval(predictions, confidence_level=95, base_volatility=0.02):
    """
    Calculate proper confidence intervals based on model ensemble variance
    and historical volatility.
    """
    if isinstance(predictions, dict):
        pred_values = list(predictions.values())
    else:
        pred_values = predictions
    
    # Ensure pred_values is always iterable
    if not hasattr(pred_values, '__len__') or isinstance(pred_values, (str, np.number)):
        pred_values = [pred_values]
    pred_values = ensure_iterable(pred_values)
    
    mean_pred = np.mean(pred_values)
    model_std = np.std(pred_values) if len(pred_values) > 1 else 0
    total_uncertainty = np.sqrt(model_std**2 + base_volatility**2)
    
    # Handle edge cases where calculations might result in NaN
    if np.isnan(mean_pred):
        mean_pred = 0.0
    if np.isnan(total_uncertainty) or total_uncertainty == 0:
        total_uncertainty = base_volatility
    
    if len(pred_values) < 30:
        alpha = (100 - confidence_level) / 100
        t_value = stats.t.ppf(1 - alpha/2, df=len(pred_values)-1)
        margin_of_error = t_value * total_uncertainty
    else:
        alpha = (100 - confidence_level) / 100
        z_value = stats.norm.ppf(1 - alpha/2)
        margin_of_error = z_value * total_uncertainty
    
    # Ensure margin_of_error is not NaN
    if np.isnan(margin_of_error):
        margin_of_error = 0.05  # Default 5% margin
    
    lower_bound = mean_pred - margin_of_error
    upper_bound = mean_pred + margin_of_error
    return lower_bound, upper_bound

def calculate_regime_adjusted_confidence(predictions, market_condition, market_strength, confidence_level=95):
    """Calculate confidence intervals based on market regime"""
    regime_volatility = {
        'bull': 0.015,
        'bear': 0.025,
        'sideways': 0.018,
        'volatile': 0.035
    }
    base_vol = regime_volatility.get(market_condition, 0.02)
    adjusted_vol = base_vol * (0.5 + market_strength)
    
    print(f"[calculate_regime_adjusted_confidence] Market: {market_condition}, strength: {market_strength}")
    print(f"[calculate_regime_adjusted_confidence] Base vol: {base_vol:.6f}, adjusted vol: {adjusted_vol:.6f}")
    
    # Convert single prediction to proper format
    if not hasattr(predictions, '__len__') or isinstance(predictions, (str, np.number)):
        prediction_value = float(predictions)
    else:
        prediction_value = np.mean(predictions) if len(predictions) > 0 else 0.0
    
    # Calculate confidence interval bounds
    alpha = (100 - confidence_level) / 100
    z_value = stats.norm.ppf(1 - alpha/2)
    
    margin_of_error = z_value * adjusted_vol
    
    print(f"[calculate_regime_adjusted_confidence] Z-value: {z_value:.3f}, margin: {margin_of_error:.6f}")
    
    lower_bound = prediction_value - margin_of_error
    upper_bound = prediction_value + margin_of_error
    
    return lower_bound, upper_bound

def calculate_historical_confidence(df, predictions, confidence_level=95, lookback_days=252, prediction_window=5):
    """Calculate confidence intervals using historical volatility"""
    if df is None or len(df) < 30:
        # Fallback to basic confidence calculation
        if not hasattr(predictions, '__len__') or isinstance(predictions, (str, np.number)):
            predictions = [predictions]
        predictions = ensure_iterable(predictions)
        return calculate_proper_confidence_interval(predictions, confidence_level)
    
    # Calculate historical volatility
    returns = df['Close'].pct_change().dropna()
    if len(returns) < 30:
        # Not enough data for meaningful volatility calculation
        return calculate_proper_confidence_interval([predictions], confidence_level)
    
    # Use available data, but at least 30 days
    lookback_data = returns.tail(min(lookback_days, len(returns)))
    historical_vol = lookback_data.std()
    
    # Scale volatility to prediction window
    scaled_vol = historical_vol * np.sqrt(prediction_window)
    
    print(f"[calculate_historical_confidence] Historical vol: {historical_vol:.6f}, scaled vol: {scaled_vol:.6f}")
    
    # Convert single prediction to proper format
    if not hasattr(predictions, '__len__') or isinstance(predictions, (str, np.number)):
        prediction_value = float(predictions)
    else:
        prediction_value = np.mean(predictions) if len(predictions) > 0 else 0.0
    
    # Calculate confidence interval bounds
    alpha = (100 - confidence_level) / 100
    z_value = stats.norm.ppf(1 - alpha/2)  # Use normal distribution for volatility-based CI
    
    margin_of_error = z_value * scaled_vol
    
    print(f"[calculate_historical_confidence] Z-value: {z_value:.3f}, margin: {margin_of_error:.6f}")
    
    lower_bound = prediction_value - margin_of_error
    upper_bound = prediction_value + margin_of_error
    
    return lower_bound, upper_bound

def fixed_single_ticker_prediction(df, model_predictions, market_condition, market_strength, confidence_level, prediction_window=5):
    """Calculate prediction and confidence intervals for a single ticker."""
    
    # Handle different formats of model_predictions
    if isinstance(model_predictions, dict):
        # Check if it's the new structured format from train_model_for_stock
        if 'prediction' in model_predictions and 'individual_predictions' in model_predictions:
            # New format: use the ensemble prediction directly
            avg_pred = model_predictions['prediction']
            print(f"[fixed_single_ticker_prediction] Using structured format prediction: {avg_pred}")
        else:
            # Old format: extract predictions from dict values
            values_list = list(model_predictions.values())
            avg_preds = []
            for v in values_list:
                if isinstance(v, dict) and 'avg_pred' in v:
                    avg_preds.append(v['avg_pred'])
                elif isinstance(v, (int, float)):
                    avg_preds.append(float(v))
                elif isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    avg_preds.append(float(v[-1]) if hasattr(v[-1], '__float__') else 0.0)
            
            avg_preds = ensure_iterable(avg_preds)
            avg_pred = np.mean([p for p in avg_preds if not np.isnan(p)]) if len(avg_preds) > 0 else 0.0
            print(f"[fixed_single_ticker_prediction] Using old format, extracted predictions: {avg_preds}, average: {avg_pred}")
    else:
        # Handle single prediction value
        avg_pred = float(model_predictions) if model_predictions is not None else 0.0
        print(f"[fixed_single_ticker_prediction] Using single prediction value: {avg_pred}")
    
    # Handle NaN in prediction
    if np.isnan(avg_pred):
        avg_pred = 0.0
        print(f"[fixed_single_ticker_prediction] Prediction was NaN, set to 0.0")
    
    print(f"[fixed_single_ticker_prediction] Final avg_pred: {avg_pred} ({avg_pred*100:.3f}%)")
    print(f"[fixed_single_ticker_prediction] Market condition: {market_condition}, strength: {market_strength}")
    print(f"[fixed_single_ticker_prediction] Historical data available: {df is not None and len(df) > 50}")
    
    if df is not None and len(df) > 50:
        print(f"[fixed_single_ticker_prediction] Using historical confidence calculation with {len(df)} data points")
        lower, upper = calculate_historical_confidence(df, avg_pred, confidence_level, prediction_window=prediction_window)
        print(f"[fixed_single_ticker_prediction] Historical confidence result: lower={lower}, upper={upper}")
    else:
        print(f"[fixed_single_ticker_prediction] Using regime-adjusted confidence calculation")
        lower, upper = calculate_regime_adjusted_confidence(avg_pred, market_condition, market_strength, confidence_level)
        print(f"[fixed_single_ticker_prediction] Regime confidence result: lower={lower}, upper={upper}")
    
    # Ensure bounds are not NaN
    if np.isnan(lower):
        print(f"[fixed_single_ticker_prediction] Lower bound was NaN, using fallback: {avg_pred - 0.05}")
        lower = avg_pred - 0.05
    if np.isnan(upper):
        print(f"[fixed_single_ticker_prediction] Upper bound was NaN, using fallback: {avg_pred + 0.05}")
        upper = avg_pred + 0.05
    
    print(f"[fixed_single_ticker_prediction] Final confidence interval: [{lower:.6f}, {upper:.6f}] = [{lower*100:.3f}%, {upper*100:.3f}%]")
    
    return {
        'prediction': float(avg_pred),
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'confidence_level': confidence_level
    }

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

# Market symbols for cross-asset features
MARKET_SYMBOLS = ['SPY', 'VIX', 'DXY', 'TLT', 'GLD', 'QQQ']
market_data_cache = {}  # Global cache for market data
CACHE_MAX_SIZE = 50  # Maximum number of cache entries to prevent memory bloat
CACHE_MAX_AGE_HOURS = 24  # Maximum age of cache entries in hours

def manage_cache_size():
    """Manage cache size to prevent memory bloat"""
    global market_data_cache
    
    if len(market_data_cache) <= CACHE_MAX_SIZE:
        return
    
    print(f"[manage_cache_size] Cache size ({len(market_data_cache)}) exceeds limit ({CACHE_MAX_SIZE}), cleaning up...")
    
    # Remove oldest entries based on access time or creation time
    # For simplicity, we'll clear half the cache when it gets too large
    cache_items = list(market_data_cache.items())
    items_to_keep = cache_items[:CACHE_MAX_SIZE // 2]
    
    market_data_cache = dict(items_to_keep)
    print(f"[manage_cache_size] Cache cleaned, new size: {len(market_data_cache)}")

def clear_stale_cache():
    """Clear cache entries older than CACHE_MAX_AGE_HOURS"""
    global market_data_cache
    
    try:
        import time
        current_time = time.time()
        stale_keys = []
        
        # Check cache files for staleness
        for cache_key in os.listdir(CACHE_DIR):
            if cache_key.endswith('.pkl'):
                cache_file = os.path.join(CACHE_DIR, cache_key)
                file_age_hours = (current_time - os.path.getmtime(cache_file)) / 3600
                
                if file_age_hours > CACHE_MAX_AGE_HOURS:
                    stale_keys.append(cache_key)
        
        # Remove stale files
        for key in stale_keys:
            try:
                os.remove(os.path.join(CACHE_DIR, key))
                print(f"[clear_stale_cache] Removed stale cache file: {key}")
            except Exception as e:
                print(f"[clear_stale_cache] Error removing {key}: {e}")
        
        if stale_keys:
            print(f"[clear_stale_cache] Cleaned {len(stale_keys)} stale cache entries")
            
    except Exception as e:
        print(f"[clear_stale_cache] Error during cache cleanup: {e}")

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
        "CME", "ICE", "MMC", "WTW", "JKHY", "BRKB", "L", "SPGI", "MA", "V", "FDS", "AON", "CB", "BRO", "CBOE", "AFL", "AJG",
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

def download_market_data_cache(start_date, force_refresh=False):
    """Download and cache market data for cross-asset features"""
    global market_data_cache
    
    # Clean up stale cache entries before downloading new data
    clear_stale_cache()
    
    cache_key = f"market_data_{start_date.replace('-', '')}"
    
    if not force_refresh:
        cached_data = get_cached_data(cache_key)
        if cached_data:
            market_data_cache = cached_data
            manage_cache_size()  # Ensure cache doesn't grow too large
            return cached_data
    
    print("[download_market_data_cache] Downloading market reference data...")
    market_data_cache = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        def fetch_market_symbol(symbol):
            try:
                df = yf.download(symbol, start=start_date, progress=False)
                if df is not None and not df.empty and len(df) >= 50:
                    return symbol, df
            except Exception as e:
                print(f"[download_market_data_cache] Error downloading {symbol}: {e}")
            return symbol, None
        
        futures = {executor.submit(fetch_market_symbol, symbol): symbol for symbol in MARKET_SYMBOLS}
        for future in as_completed(futures):
            symbol, data = future.result()
            if data is not None:
                market_data_cache[symbol] = data
    
    # Save to cache and manage cache size
    save_to_cache(cache_key, market_data_cache)
    manage_cache_size()  # Ensure cache doesn't grow too large
    print(f"[download_market_data_cache] Successfully cached {len(market_data_cache)} market symbols")
    return market_data_cache

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
    # Ensure x is iterable and has length
    if not hasattr(x, '__len__') or isinstance(x, (str, np.number)):
        x = [x]
    x = ensure_iterable(x)
    
    if len(x) < 2:
        return np.nan
    return np.polyfit(range(len(x)), x, 1)[0]

def add_features_to_stock_original(ticker, df, prediction_window=5):
    """Original feature engineering function (backup)"""
    if df is None or len(df) < 50:
        print(f"[add_features_to_stock_original] {ticker}: DataFrame is None or too short ({len(df) if df is not None else 0} rows)")
        return None
    try:
        df = df.copy()
        # Convert price series to float and handle missing data
        try:
            for col in ['Close', 'High', 'Low', 'Volume']:
                if col not in df.columns:
                    print(f"[add_features_to_stock_original] {ticker}: Missing required column {col}")
                    return None
                
                # Handle potential MultiIndex columns or Series data
                series_data = df[col]
                if hasattr(series_data, 'values'):
                    series_data = series_data.values
                
                # Flatten the series if it's nested
                if isinstance(series_data, (list, tuple)):
                    series_data = np.array(series_data).flatten()
                elif hasattr(series_data, 'flatten'):
                    series_data = series_data.flatten()
                
                # Convert to numeric Series
                df[f'{col.lower()}_series'] = pd.Series(
                    pd.to_numeric(series_data, errors='coerce'), 
                    index=df.index
                )
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error converting price data: {e}")
            return None

        # Initialize features list to track successful calculations
        calculated_features = []

        # --- Momentum ---
        try:
            rsi = RSIIndicator(close=df['close_series'])
            df['RSI'] = pd.Series(rsi.rsi(), index=df.index)
            if 'RSI' in df.columns and not df['RSI'].isna().all():
                calculated_features.append('RSI')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating RSI: {e}")

        try:
            df['roc_10'] = flatten_series(ROCIndicator(close=df['close_series']).roc())
            if 'roc_10' in df.columns and not df['roc_10'].isna().all():
                calculated_features.append('roc_10')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating ROC: {e}")

        try:
            df['momentum'] = flatten_series(df['close_series'] - df['close_series'].shift(10))
            if 'momentum' in df.columns and not df['momentum'].isna().all():
                calculated_features.append('momentum')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating momentum: {e}")

        try:
            if 'RSI' in df.columns:
                df['rsi_14_diff'] = flatten_series(df['RSI'] - df['RSI'].shift(5))
                if 'rsi_14_diff' in df.columns and not df['rsi_14_diff'].isna().all():
                    calculated_features.append('rsi_14_diff')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating RSI difference: {e}")

        # --- Trend ---
        try:
            macd = MACD(close=df['close_series'])
            df['macd'] = flatten_series(macd.macd().squeeze())
            df['macd_signal'] = flatten_series(macd.macd_signal().squeeze())
            df['ema_diff'] = flatten_series((df['macd'] - df['macd_signal']).squeeze())
            calculated_features.extend(['macd', 'macd_signal', 'ema_diff'])
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating MACD: {e}")

        try:
            df['slope_price_10d'] = flatten_series(df['close_series'].rolling(window=10).apply(calc_slope, raw=False))
            calculated_features.append('slope_price_10d')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating slope of price: {e}")

        try:
            ema_10 = flatten_series(EMAIndicator(close=df['close_series'], window=10).ema_indicator())
            df['ema_ratio'] = flatten_series(df['close_series'] / ema_10)
            calculated_features.append('ema_ratio')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating EMA ratio: {e}")

        try:
            for col in ['macd', 'macd_signal', 'ema_diff']:
                series = df[col]
                df[col] = flatten_series((series - series.mean()) / (series.std() if series.std() != 0 else 1))
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error normalizing MACD features: {e}")

        # --- Momentum ---
        try:
            stoch = StochasticOscillator(high=df['high_series'], low=df['low_series'], close=df['close_series'])
            df['stoch_k'] = flatten_series(stoch.stoch().squeeze())
            if df['stoch_k'].max() != df['stoch_k'].min():
                df['stoch_k'] = flatten_series(100 * (df['stoch_k'] - df['stoch_k'].min()) / (df['stoch_k'].max() - df['stoch_k'].min()))
            else:
                df['stoch_k'] = flatten_series(50)  # fallback if constant
            calculated_features.append('stoch_k')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating Stochastic Oscillator: {e}")

        # --- Volatility ---
        try:
            df['ATR'] = flatten_series(AverageTrueRange(high=df['high_series'], low=df['low_series'], close=df['close_series']).average_true_range())
            calculated_features.append('ATR')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating ATR: {e}")

        try:
            df['volatility'] = flatten_series(np.log(df['close_series'] / df['close_series'].shift(1)).rolling(10).std())
            calculated_features.append('volatility')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating volatility: {e}")

        try:
            df['rolling_20d_std'] = flatten_series(df['close_series'].rolling(window=20).std())
            calculated_features.append('rolling_20d_std')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating rolling standard deviation: {e}")

        try:
            bb = BollingerBands(close=df['close_series'])
            df['bb_width'] = flatten_series(bb.bollinger_wband())
            calculated_features.append('bb_width')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating Bollinger Band width: {e}")

        try:
            df['donchian_width'] = flatten_series(df['high_series'].rolling(window=20).max() - df['low_series'].rolling(window=20).min())
            calculated_features.append('donchian_width')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating Donchian width: {e}")

        # --- Volume ---
        try:
            df['OBV'] = flatten_series(OnBalanceVolumeIndicator(close=df['close_series'], volume=df['volume_series']).on_balance_volume())
            calculated_features.append('OBV')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating OBV: {e}")

        try:
            df['volume_pct_change'] = flatten_series(df['Volume'].pct_change())
            calculated_features.append('volume_pct_change')
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating volume percentage change: {e}")

        # --- Other/Composite ---
        try:
            df['close_lag_1'] = flatten_series(df['Close'].shift(1))
            df['close_lag_5'] = flatten_series(df['Close'].shift(5))
            df['past_10d_return'] = flatten_series(df['close_series'] / df['close_series'].shift(10) - 1)
            df[f'close_lead_{prediction_window}'] = flatten_series(df['Close'].shift(-prediction_window))
            df[f'forward_return_{prediction_window}'] = flatten_series((df['Close'].shift(-prediction_window) / df['Close']) - 1)
            df[f'rolling_max_{prediction_window}'] = flatten_series(df['Close'].rolling(window=prediction_window).max())
            df[f'rolling_min_{prediction_window}'] = flatten_series(df['Close'].rolling(window=prediction_window).min())
            calculated_features.extend(['close_lag_1', 'close_lag_5', 'past_10d_return', f'close_lead_{prediction_window}', f'forward_return_{prediction_window}', f'rolling_max_{prediction_window}', f'rolling_min_{prediction_window}'])
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating composite features: {e}")

        try:
            ichimoku = IchimokuIndicator(high=df['high_series'], low=df['low_series'], window1=9, window2=26, window3=52, fillna=True)
            df['ichimoku_a'] = flatten_series(ichimoku.ichimoku_a())
            df['ichimoku_b'] = flatten_series(ichimoku.ichimoku_b())
            df['ichimoku_base'] = flatten_series(ichimoku.ichimoku_base_line())
            calculated_features.extend(['ichimoku_a', 'ichimoku_b', 'ichimoku_base'])
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating Ichimoku features: {e}")

        # Drop rows with NaN for rolling features, but NOT for close_lead_N, forward_return_N, rolling_max/min
        # Only include columns that actually exist in the DataFrame
        existing_feature_cols = [col for col in calculated_features if col in df.columns]

        # Add lagged features for selected indicators (only if they exist)
        lag_features = ['RSI', 'macd', 'stoch_k', 'ATR', 'rolling_20d_std', 'percent_b', 'OBV', 'cmf']
        num_lags = 3
        for feat in lag_features:
            for lag in range(1, num_lags+1):
                col_name = f'{feat}_lag{lag}'
                if col_name in df.columns:
                    existing_feature_cols.append(col_name)
        
        if existing_feature_cols:
            df = df.dropna(subset=existing_feature_cols)
        else:
            print(f"[add_features_to_stock_original] {ticker}: No valid features calculated")
            return None
            
        return df
    except Exception as e:
        print(f"[add_features_to_stock_original] ERROR for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_features_to_stock(ticker, df, prediction_window=5, market_data_cache=None):
    """Enhanced version with cross-asset features"""
    if df is None or len(df) < 50:
        print(f"[add_features_to_stock] {ticker}: DataFrame too short or None")
        return None
    
    try:
        # Start with existing technical features
        df = add_features_to_stock_original(ticker, df, prediction_window)
        if df is None:
            return None
        
        # Add cross-asset features if market data is available
        if market_data_cache:
            if not isinstance(market_data_cache, dict) or len(market_data_cache) == 0:
                print(f"[add_features_to_stock] {ticker}: Market data cache is empty or invalid, skipping cross-asset features")
            else:
                print(f"[add_features_to_stock] {ticker}: Adding cross-asset features using {len(market_data_cache)} market symbols")
                try:
                    df = add_cross_asset_features(df, ticker, market_data_cache)
                except Exception as e:
                    print(f"[add_features_to_stock] {ticker}: Error adding cross-asset features: {e}")
                    # Continue without cross-asset features rather than failing completely
        else:
            print(f"[add_features_to_stock] {ticker}: No market data cache provided, using technical features only")
        
        return df
        
    except Exception as e:
        print(f"[add_features_to_stock] ERROR for {ticker}: {e}")
        return None

def add_cross_asset_features(df, ticker, market_data_cache):
    """Add cross-asset correlation and macro features"""
    try:
        calculated_features = []
        
        # Validate market data cache
        if not market_data_cache or not isinstance(market_data_cache, dict):
            print(f"[add_cross_asset_features] {ticker}: Invalid market data cache")
            return df
        
        available_symbols = list(market_data_cache.keys())
        print(f"[add_cross_asset_features] {ticker}: Available market symbols: {available_symbols}")
        
        # Market Relative Performance (SPY correlation)
        if 'SPY' in market_data_cache:
            try:
                # Limit forward fill to 5 days to prevent stale data propagation
                spy_data = market_data_cache['SPY']['Close'].reindex(df.index, method='ffill', limit=5)
                if len(spy_data.dropna()) > 20:
                    spy_returns = spy_data.pct_change()
                    stock_returns = df['Close'].pct_change()
                    
                    # Relative strength vs market
                    df['relative_strength'] = flatten_series(stock_returns - spy_returns)
                    
                    # Rolling correlation with market
                    df['correlation_spy_20'] = flatten_series(
                        stock_returns.rolling(20).corr(spy_returns)
                    )
                    
                    # Beta calculation (20-day rolling)
                    df['beta_spy_20'] = flatten_series(
                        stock_returns.rolling(20).cov(spy_returns) / spy_returns.rolling(20).var()
                    )
                    
                    # Relative performance vs SPY (price ratio)
                    df['spy_ratio'] = flatten_series(df['Close'] / spy_data)
                    df['spy_ratio_ma'] = flatten_series(df['spy_ratio'].rolling(20).mean())
                    
                    calculated_features.extend([
                        'relative_strength', 'correlation_spy_20', 'beta_spy_20', 
                        'spy_ratio', 'spy_ratio_ma'
                    ])
            except Exception as e:
                print(f"[add_cross_asset_features] {ticker}: Error calculating SPY features: {e}")
        
        # VIX Features (Fear Index)
        if 'VIX' in market_data_cache:
            # Limit forward fill to 3 days for VIX (fear index should be current)
            vix_data = market_data_cache['VIX']['Close'].reindex(df.index, method='ffill', limit=3)
            if len(vix_data.dropna()) > 20:
                df['vix_level'] = flatten_series(vix_data)
                df['vix_change'] = flatten_series(vix_data.pct_change())
                df['vix_vs_ma'] = flatten_series(vix_data / vix_data.rolling(20).mean())
                
                # VIX regime indicators
                df['vix_high'] = flatten_series((vix_data > 25).astype(int))  # High fear
                df['vix_low'] = flatten_series((vix_data < 15).astype(int))   # Low fear
                
                calculated_features.extend([
                    'vix_level', 'vix_change', 'vix_vs_ma', 'vix_high', 'vix_low'
                ])
        
        # Dollar Index (DXY) Features
        if 'DXY' in market_data_cache:
            # Limit forward fill to 7 days for DXY (currency moves slower)
            dxy_data = market_data_cache['DXY']['Close'].reindex(df.index, method='ffill', limit=7)
            if len(dxy_data.dropna()) > 20:
                dxy_returns = dxy_data.pct_change()
                stock_returns = df['Close'].pct_change()
                
                df['dxy_correlation'] = flatten_series(
                    stock_returns.rolling(20).corr(dxy_returns)
                )
                df['dxy_level'] = flatten_series(dxy_data)
                df['dxy_strength'] = flatten_series(
                    (dxy_data > dxy_data.rolling(50).mean()).astype(int)
                )
                
                calculated_features.extend(['dxy_correlation', 'dxy_level', 'dxy_strength'])
        
        # Treasury Bonds (TLT) - Interest Rate Proxy
        if 'TLT' in market_data_cache:
            # Limit forward fill to 5 days for bonds
            tlt_data = market_data_cache['TLT']['Close'].reindex(df.index, method='ffill', limit=5)
            if len(tlt_data.dropna()) > 20:
                tlt_returns = tlt_data.pct_change()
                stock_returns = df['Close'].pct_change()
                
                df['tlt_correlation'] = flatten_series(
                    stock_returns.rolling(20).corr(tlt_returns)
                )
                df['tlt_trend'] = flatten_series(
                    (tlt_data > tlt_data.rolling(20).mean()).astype(int)
                )
                
                calculated_features.extend(['tlt_correlation', 'tlt_trend'])
        
        # Gold (GLD) - Risk-off Asset
        if 'GLD' in market_data_cache:
            # Limit forward fill to 5 days for gold
            gld_data = market_data_cache['GLD']['Close'].reindex(df.index, method='ffill', limit=5)
            if len(gld_data.dropna()) > 20:
                gld_returns = gld_data.pct_change()
                stock_returns = df['Close'].pct_change()
                
                df['gld_correlation'] = flatten_series(
                    stock_returns.rolling(20).corr(gld_returns)
                )
                
                calculated_features.append('gld_correlation')
        
        # Tech vs Market (QQQ vs SPY)
        if 'QQQ' in market_data_cache and 'SPY' in market_data_cache:
            # Limit forward fill to 5 days for both tech and market indices
            qqq_data = market_data_cache['QQQ']['Close'].reindex(df.index, method='ffill', limit=5)
            spy_data = market_data_cache['SPY']['Close'].reindex(df.index, method='ffill', limit=5)
            
            if len(qqq_data.dropna()) > 20 and len(spy_data.dropna()) > 20:
                df['qqq_spy_ratio'] = flatten_series(qqq_data / spy_data)
                df['tech_outperform'] = flatten_series(
                    (df['qqq_spy_ratio'] > df['qqq_spy_ratio'].rolling(20).mean()).astype(int)
                )
                
                calculated_features.extend(['qqq_spy_ratio', 'tech_outperform'])
        
        # Enhanced Time-Based Features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = (df.index == df.index.to_period('M').end_time).astype(int)
        df['is_quarter_end'] = (df.index == df.index.to_period('Q').end_time).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        
        calculated_features.extend([
            'day_of_week', 'month', 'quarter', 'is_month_end', 
            'is_quarter_end', 'is_friday', 'is_monday'
        ])
        
        # Market Regime Features (based on multiple assets)
        regime_features = calculate_market_regime_features(market_data_cache, df.index)
        for feature_name, feature_values in regime_features.items():
            df[feature_name] = flatten_series(feature_values)
            calculated_features.append(feature_name)
        
        print(f"[add_cross_asset_features] {ticker}: Added {len(calculated_features)} cross-asset features")
        return df
        
    except Exception as e:
        print(f"[add_cross_asset_features] Error for {ticker}: {e}")
        return df

def calculate_market_regime_features(market_data_cache, target_index):
    """Calculate regime features based on multiple market indicators"""
    regime_features = {}
    
    try:
        # Multi-asset trend strength
        if 'SPY' in market_data_cache and 'VIX' in market_data_cache:
            # Limit forward fill to prevent stale regime data
            spy_data = market_data_cache['SPY']['Close'].reindex(target_index, method='ffill', limit=5)
            vix_data = market_data_cache['VIX']['Close'].reindex(target_index, method='ffill', limit=3)
            
            # Risk-on/Risk-off regime
            spy_trend = (spy_data > spy_data.rolling(20).mean()).astype(int)
            vix_trend = (vix_data < vix_data.rolling(20).mean()).astype(int)
            
            regime_features['risk_on'] = spy_trend & vix_trend  # SPY up, VIX down
            regime_features['risk_off'] = (~spy_trend) & (~vix_trend)  # SPY down, VIX up
        
        # Interest rate environment
        if 'TLT' in market_data_cache:
            tlt_data = market_data_cache['TLT']['Close'].reindex(target_index, method='ffill', limit=5)
            regime_features['rising_rates'] = (tlt_data < tlt_data.rolling(50).mean()).astype(int)
        
        # Dollar strength regime
        if 'DXY' in market_data_cache:
            dxy_data = market_data_cache['DXY']['Close'].reindex(target_index, method='ffill', limit=7)
            regime_features['strong_dollar'] = (dxy_data > dxy_data.rolling(50).mean()).astype(int)
        
    except Exception as e:
        print(f"[calculate_market_regime_features] Error: {e}")
    
    return regime_features

def add_features_parallel(stock_data, prediction_window=5, market_data_cache=None):
    """Add features to all stocks using parallel processing, with N-day lookahead."""
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_ticker = {
            executor.submit(add_features_to_stock, ticker, df, prediction_window, market_data_cache): ticker 
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
    # Define base weights for each group per regime - updated with cross_asset group
    base_weights = {
        'bull': {'momentum': 1.3, 'trend': 1.2, 'mean_reversion': 0.8, 'volatility': 0.7, 'volume': 1.0, 'cross_asset': 1.1, 'other': 1.0},
        'bear': {'momentum': 0.8, 'trend': 0.7, 'mean_reversion': 1.3, 'volatility': 1.2, 'volume': 1.0, 'cross_asset': 1.2, 'other': 1.0},
        'sideways': {'momentum': 0.8, 'trend': 0.8, 'mean_reversion': 1.3, 'volatility': 1.0, 'volume': 1.0, 'cross_asset': 0.9, 'other': 1.0},
        'volatile': {'momentum': 0.7, 'trend': 0.7, 'mean_reversion': 1.1, 'volatility': 1.3, 'volume': 1.2, 'cross_asset': 1.3, 'other': 1.0},
    }
    # Map each feature to a group - updated with cross-asset features
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
        
        # NEW: Cross-asset features
        'relative_strength': 'cross_asset', 'correlation_spy_20': 'cross_asset', 
        'beta_spy_20': 'cross_asset', 'spy_ratio': 'cross_asset', 'spy_ratio_ma': 'cross_asset',
        'vix_level': 'cross_asset', 'vix_change': 'cross_asset', 'vix_vs_ma': 'cross_asset',
        'vix_high': 'cross_asset', 'vix_low': 'cross_asset',
        'dxy_correlation': 'cross_asset', 'dxy_level': 'cross_asset', 'dxy_strength': 'cross_asset',
        'tlt_correlation': 'cross_asset', 'tlt_trend': 'cross_asset',
        'gld_correlation': 'cross_asset',
        'qqq_spy_ratio': 'cross_asset', 'tech_outperform': 'cross_asset',
        'risk_on': 'cross_asset', 'risk_off': 'cross_asset', 
        'rising_rates': 'cross_asset', 'strong_dollar': 'cross_asset',
        
        # Enhanced time features
        'day_of_week': 'other', 'month': 'other', 'quarter': 'other', 
        'is_month_end': 'other', 'is_quarter_end': 'other', 
        'is_friday': 'other', 'is_monday': 'other'
    }
    # Add lagged features to group_map
    lag_features = {
        'RSI': 'momentum', 'macd': 'trend', 'stoch_k': 'momentum', 'ATR': 'volatility',
        'rolling_20d_std': 'volatility', 'percent_b': 'mean_reversion', 'OBV': 'volume', 'cmf': 'volume'
    }
    num_lags = 3
    for base, group in lag_features.items():
        for lag in range(1, num_lags+1):
            group_map[f'{base}_lag{lag}'] = group
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
        print(f"[train_model_for_stock] {ticker}: DataFrame too short or None")
        return None
    
    try:
        # Extract feature columns (exclude basic OHLCV)
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        # Validate that we have meaningful features
        cross_asset_features = [col for col in feature_columns if any(keyword in col for keyword in 
                               ['spy', 'vix', 'dxy', 'tlt', 'gld', 'qqq', 'relative_strength', 'correlation', 'beta', 'risk_on', 'risk_off'])]
        technical_features = [col for col in feature_columns if col not in cross_asset_features]
        
        print(f"[train_model_for_stock] {ticker}: Technical features: {len(technical_features)}, Cross-asset features: {len(cross_asset_features)}")
        
        if len(technical_features) < 5:
            print(f"[train_model_for_stock] {ticker}: Insufficient technical features ({len(technical_features)})")
            return None
        
        # Clean and prepare features
        df_clean = df.copy()
        for col in feature_columns:
            if col in df_clean.columns:
                # Handle any nested arrays or objects
                series = df_clean[col]
                if series.dtype == 'object':
                    series = series.apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) > 0 else x)
                df_clean[col] = pd.to_numeric(series, errors='coerce')
        
        # Remove columns that are all NaN and validate feature quality
        valid_features = []
        failed_features = []
        for col in feature_columns:
            if col in df_clean.columns:
                if df_clean[col].isna().all():
                    failed_features.append(col)
                elif df_clean[col].var() == 0:  # No variance (constant values)
                    failed_features.append(f"{col} (constant)")
                else:
                    valid_features.append(col)
            else:
                failed_features.append(f"{col} (missing)")
        
        # Report feature validation results
        if failed_features:
            print(f"[train_model_for_stock] {ticker}: Failed features ({len(failed_features)}): {failed_features[:10]}{'...' if len(failed_features) > 10 else ''}")
        
        # Categorize valid features for better reporting
        valid_technical = [col for col in valid_features if col in technical_features]
        valid_cross_asset = [col for col in valid_features if col in cross_asset_features]
        
        print(f"[train_model_for_stock] {ticker}: Valid features - Technical: {len(valid_technical)}, Cross-asset: {len(valid_cross_asset)}")
        
        if len(valid_features) < 5:
            print(f"[train_model_for_stock] {ticker}: Not enough valid features ({len(valid_features)})")
            return None
        
        # Prepare feature matrix
        X = df_clean[valid_features].values.astype(float)
        
        # Apply regime-based feature weights if provided
        if regime is not None:
            weights_dict = get_indicator_weights(regime, regime_strength)
            weights = np.array([weights_dict.get(col, 1.0) for col in valid_features])
            X = X * weights
        
        # Prepare target: future returns
        # Use simple 1-day forward return as primary target
        future_returns = df_clean['Close'].pct_change().shift(-1).values
        
        # DO NOT scale - keep returns in their natural scale for accuracy
        # Small returns (0.001 = 0.1%) are normal and should be preserved
        
        # Remove rows with NaN in features or target
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(future_returns))
        X_clean = X[valid_mask]
        y_clean = future_returns[valid_mask]
        
        if len(X_clean) < 30:
            print(f"[train_model_for_stock] {ticker}: Not enough clean data ({len(X_clean)} rows)")
            return None
        
        # Print some debugging info about target distribution
        y_std = np.std(y_clean)
        y_mean = np.mean(y_clean)
        print(f"[train_model_for_stock] {ticker}: Target stats - Mean: {y_mean:.6f} ({y_mean*100:.3f}%), Std: {y_std:.6f} ({y_std*100:.3f}%)")
        
        # Train models and get predictions
        model_predictions = {}
        for model_id in model_ids:
            try:
                pred_result = train_and_predict_model(X_clean, y_clean, model_id)
                if pred_result is not None and len(pred_result) > 0:
                    # Get the last prediction (most recent)
                    if isinstance(pred_result, list):
                        raw_pred = float(pred_result[-1]) if len(pred_result) > 0 else 0.0
                    else:
                        raw_pred = float(pred_result)
                    
                    # NO SCALING - keep predictions in natural decimal scale
                    avg_pred = raw_pred
                    
                    # Conservative clamping to reasonable range (±20% max)
                    avg_pred = np.clip(avg_pred, -0.20, 0.20)
                    
                    model_predictions[model_id] = {
                        'avg_pred': avg_pred,
                        'all_preds': pred_result if isinstance(pred_result, list) else [avg_pred]
                    }
                    
                    print(f"[train_model_for_stock] {ticker}: Model {model_id} prediction: {avg_pred:.6f} ({avg_pred*100:.3f}%)")
                else:
                    print(f"[train_model_for_stock] {ticker}: Model {model_id} returned None")
                    model_predictions[model_id] = {'avg_pred': 0.0, 'all_preds': [0.0]}
                    
            except Exception as e:
                print(f"[train_model_for_stock] {ticker}: Error with model {model_id}: {e}")
                model_predictions[model_id] = {'avg_pred': 0.0, 'all_preds': [0.0]}
        
        # Calculate ensemble prediction with confidence scoring
        if model_predictions:
            valid_predictions = [pred['avg_pred'] for pred in model_predictions.values() if abs(pred['avg_pred']) > 1e-6]
            
            if valid_predictions:
                # Use median for robustness against outliers
                ensemble_pred = np.median(valid_predictions)
                
                # Calculate prediction spread as confidence metric
                pred_std = np.std(valid_predictions) if len(valid_predictions) > 1 else 0.0
                pred_spread = np.max(valid_predictions) - np.min(valid_predictions) if len(valid_predictions) > 1 else 0.0
                
                # Conservative confidence scoring (lower confidence = more conservative prediction)
                confidence = max(0.1, 1.0 - (pred_spread * 10))  # High spread = low confidence
                
                # Apply confidence damping to reduce overconfident predictions
                ensemble_pred = ensemble_pred * confidence
                
                # Apply prediction bounds validation
                validated_pred, validation_warnings = validate_prediction_bounds(
                    ensemble_pred, ticker, df, strict_mode=True
                )
                
                print(f"[train_model_for_stock] {ticker}: Raw ensemble: {ensemble_pred:.6f} ({ensemble_pred*100:.3f}%)")
                print(f"[train_model_for_stock] {ticker}: Validated: {validated_pred:.6f} ({validated_pred*100:.3f}%)")
                print(f"[train_model_for_stock] {ticker}: Confidence: {confidence:.3f}, Spread: {pred_spread:.6f}")
                if validation_warnings:
                    print(f"[train_model_for_stock] {ticker}: Validation warnings: {len(validation_warnings)}")
                
                return {
                    'prediction': float(validated_pred),
                    'percentage': float(validated_pred * 100),
                    'confidence': float(confidence),
                    'individual_predictions': {k: v['avg_pred'] for k, v in model_predictions.items()},
                    'valid_models': len(valid_predictions),
                    'total_models': len(model_predictions),
                    'validation_applied': len(validation_warnings) > 0,
                    'validation_warnings': validation_warnings
                }
            else:
                print(f"[train_model_for_stock] {ticker}: No valid predictions found")
                return {
                    'prediction': 0.0,
                    'percentage': 0.0,
                    'confidence': 0.0,
                    'individual_predictions': {k: v['avg_pred'] for k, v in model_predictions.items()},
                    'valid_models': 0,
                    'total_models': len(model_predictions)
                }
        else:
            print(f"[train_model_for_stock] {ticker}: No model predictions generated")
            return {
                'prediction': 0.0,
                'percentage': 0.0,
                'confidence': 0.0,
                'individual_predictions': {},
                'valid_models': 0,
                'total_models': 0
            }
        
        print(f"[train_model_for_stock] {ticker}: Completed with {len(model_predictions)} models, features: {len(valid_features)}, samples: {len(X_clean)}")
        return model_predictions
        
    except Exception as e:
        print(f"[train_model_for_stock] ERROR for {ticker}: {e}")
        import traceback
        traceback.print_exc()
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

# Utility to ensure prediction is always a list
def ensure_list(pred):
    if pred is None:
        return [0.0]
    if isinstance(pred, (list, np.ndarray)):
        return list(pred)
    try:
        return [float(pred)]
    except Exception:
        return [0.0]

def train_and_predict_model(X, y, model_id):
    """Train actual ML model and make prediction, with feature scaling."""
    if len(X) < 50:
        print(f"[train_and_predict_model] Not enough data for model {model_id}: {len(X)} samples")
        return [0.0]
    
    try:
        X = np.array(X)
        y = np.array(y)
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 20:
            print(f"[train_and_predict_model] Not enough clean data for model {model_id}: {len(X)} samples")
            return [0.0]
        
        # Split data: use 80% for training, 20% for out-of-sample prediction
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        if len(X_train) < 10:
            print(f"[train_and_predict_model] Not enough training data for model {model_id}: {len(X_train)} samples")
            return [0.0]
        
        # Add some regularization by removing extreme outliers in target
        y_train_q25 = np.percentile(y_train, 25)
        y_train_q75 = np.percentile(y_train, 75)
        iqr = y_train_q75 - y_train_q25
        outlier_threshold = 3.0 * iqr
        
        outlier_mask = (y_train >= y_train_q25 - outlier_threshold) & (y_train <= y_train_q75 + outlier_threshold)
        X_train_clean = X_train[outlier_mask]
        y_train_clean = y_train[outlier_mask]
        
        if len(X_train_clean) < 5:
            # If too many outliers removed, use original data
            X_train_clean = X_train
            y_train_clean = y_train
        
        # Choose scaler based on model type
        if model_id in [3, 6, 7, 9, 10]:  # Neural nets, SVR, Bayesian, Elastic Net, Transformer
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Fit scaler and transform data
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_train_scaled[-1:].copy()
        
        # Initialize model based on model_id with CONSERVATIVE parameters for accuracy
        model = None
        if model_id == 1:  # XGBoost - More conservative
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=150,  # More trees for stability
                max_depth=4,       # Shallower trees to prevent overfitting
                learning_rate=0.03,  # Lower learning rate
                subsample=0.7,     # More aggressive subsampling
                colsample_bytree=0.7,
                reg_alpha=0.1,     # L1 regularization
                reg_lambda=0.1,    # L2 regularization
                random_state=42,
                verbosity=0
            )
        elif model_id == 2:  # Random Forest - More conservative
            model = RandomForestRegressor(
                n_estimators=150,  # More trees
                max_depth=8,       # Shallower depth
                min_samples_split=10,  # Higher minimum split
                min_samples_leaf=5,    # Higher minimum leaf
                bootstrap=True,
                max_features=0.7,  # Use fewer features per tree
                random_state=42,
                n_jobs=1
            )
        elif model_id == 3:  # Neural Network - Smaller and regularized
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),  # Smaller network
                activation='relu',
                solver='adam',
                alpha=0.01,        # Stronger regularization
                learning_rate_init=0.0005,  # Lower learning rate
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        elif model_id == 4:  # Extra Trees - More conservative
            model = ExtraTreesRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                bootstrap=True,
                max_features=0.7,
                random_state=42,
                n_jobs=1
            )
        elif model_id == 5:  # AdaBoost - More conservative
            model = AdaBoostRegressor(
                n_estimators=80,   # Fewer estimators
                learning_rate=0.05,  # Lower learning rate
                loss='linear',
                random_state=42
            )
        elif model_id == 6:  # Bayesian Ridge - More regularized
            model = BayesianRidge(
                alpha_1=1e-5,      # Stronger prior
                alpha_2=1e-5,
                lambda_1=1e-5,
                lambda_2=1e-5,
                fit_intercept=True,
                compute_score=True
            )
        elif model_id == 7:  # SVR
            model = SVR(
                kernel='rbf',
                C=10.0,
                epsilon=0.1,
                gamma='scale'
            )
        elif model_id == 8:  # Gradient Boosting
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
        elif model_id == 9:  # Elastic Net
            model = ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            )
        elif model_id == 10:  # Transformer-like
            model = TransformerModel(input_dim=X_train_scaled.shape[1])
        else:
            print(f"[train_and_predict_model] Unknown model_id: {model_id}")
            return [0.0]
        
        # Add cross-validation for better accuracy assessment
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        
        try:
            # Use TimeSeriesSplit for financial data (respects temporal order)
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_train_scaled)//10), test_size=max(10, len(X_train_scaled)//10))
            
            # Perform cross-validation to assess model quality
            cv_scores = cross_val_score(model, X_train_scaled, y_train_clean, cv=tscv, scoring='neg_mean_squared_error')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            print(f"[train_and_predict_model] Model {model_id} CV Score: {cv_mean:.6f} ± {cv_std:.6f}")
            
            # Skip models with very poor cross-validation performance
            if cv_mean < -0.01:  # If MSE is very high (>1% error), skip this model
                print(f"[train_and_predict_model] Model {model_id} has poor CV performance, skipping")
                return [0.0]
                
        except Exception as e:
            print(f"[train_and_predict_model] CV error for model {model_id}: {e}")
        
        # Train the model
        model.fit(X_train_scaled, y_train_clean)
        
        # Make predictions on test set (last 20% of data)
        if len(X_test_scaled) > 0:
            predictions = model.predict(X_test_scaled)
            predictions = ensure_iterable(predictions)
            
            # Apply prediction calibration/adjustment based on historical performance
            # Calculate the bias between predictions and actual values on test set
            if len(predictions) == len(y_test):
                prediction_bias = np.mean(predictions) - np.mean(y_test)
                
                # Apply bias correction to reduce systematic errors
                calibrated_predictions = [p - prediction_bias for p in predictions]
                
                # Also apply shrinkage towards zero to reduce overconfidence
                shrinkage_factor = 0.7  # Conservative shrinkage
                final_predictions = [p * shrinkage_factor for p in calibrated_predictions]
                
                # Apply bounds checking to individual predictions
                validated_predictions = []
                for i, pred in enumerate(final_predictions):
                    validated_pred, warnings = validate_prediction_bounds(pred, f"Model_{model_id}", strict_mode=False)
                    validated_predictions.append(validated_pred)
                    if warnings and i < 3:  # Only log warnings for first 3 predictions to avoid spam
                        print(f"[train_and_predict_model] Model {model_id} prediction {i}: {warnings[0] if warnings else 'OK'}")
                
                print(f"[train_and_predict_model] Model {model_id} bias correction: {prediction_bias:.6f}, validated predictions: {validated_predictions[-3:]}")
                return validated_predictions
            else:
                # Just apply shrinkage if we can't calculate bias
                shrinkage_factor = 0.7
                final_predictions = [p * shrinkage_factor for p in predictions]
                
                # Apply bounds checking to individual predictions
                validated_predictions = []
                for pred in final_predictions:
                    validated_pred, warnings = validate_prediction_bounds(pred, f"Model_{model_id}", strict_mode=False)
                    validated_predictions.append(validated_pred)
                
                return validated_predictions
            
        else:
            # If no test data, predict on last training sample
            last_sample = X_train_scaled[-1:] 
            prediction = model.predict(last_sample)
            pred_list = ensure_iterable(prediction).tolist()
            
            # Apply bounds checking to single prediction
            if pred_list:
                validated_pred, warnings = validate_prediction_bounds(pred_list[0], f"Model_{model_id}", strict_mode=False)
                pred_list = [validated_pred]
            
            print(f"[train_and_predict_model] Model {model_id} single prediction: {pred_list}")
            return pred_list
        
    except Exception as e:
        print(f"[train_and_predict_model] Error training model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return [0.0]

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
    # Add market condition influence
    market_condition, _ = calculate_market_condition(df)
    if market_condition == 'bull':
        base_prediction += 0.01
    elif market_condition == 'bear':
        base_prediction -= 0.01
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
        pred_start = hist_norm.iloc[-1]
        pred_end = pred_start * (1 + stock['pred'])  # Total return over prediction window
        
        # Create linear interpolation from current price to predicted price
        pred_path = []
        for i in range(prediction_window + 1):
            # Linear interpolation: start + (end - start) * (i / prediction_window)
            progress = i / prediction_window if prediction_window > 0 else 0
            pred_value = pred_start + (pred_end - pred_start) * progress
            pred_path.append(pred_value)
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
    # Get show_type from the current scope if available, default to 'top'
    show_type = getattr(plt, '_show_type', 'top') if hasattr(plt, '_show_type') else 'top'
    title_prefix = "Top" if show_type == 'top' else "Worst"
    plt.title(f"{title_prefix} {len(stock_predictions)} Stock Predictions (window={prediction_window} days)", color='white', fontsize=16)
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

def add_features_single(ticker, df, prediction_window=5, market_data_cache=None):
    """Add features to a single ticker's data (no parallelization)."""
    return add_features_to_stock(ticker, df, prediction_window, market_data_cache)

def train_model_single(df, model_ids, regime=None, regime_strength=0.5):
    """Train models for a single ticker (no parallelization) with bounds checking."""
    if df is None or len(df) < 50:
        print(f"[train_model_single] DataFrame too short or None")
        return None
    
    try:
        # Use the same logic as train_model_for_stock but for single ticker
        return train_model_for_stock("SINGLE_TICKER", df, model_ids, regime, regime_strength)
        
    except Exception as e:
        print(f"[train_model_single] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_single_ticker_chart(df, predictions, prediction_window):
    """Create a chart for a single ticker: last X days of history and X days of forecast, normalized to 100 at -X."""
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    if df is None or len(df) < prediction_window + 1:
        return None
    
    # Get history data - last prediction_window + 1 days so we have prediction_window days of history plus current
    hist = df['Close'].iloc[-(prediction_window+1):]
    norm_factor = hist.iloc[0]  # First value becomes 100
    hist_norm = hist / norm_factor * 100
    
    # DEBUG: Print prediction values to understand what we're getting
    print(f"[predict_single_ticker_chart] Raw predictions: {predictions}")
    
    # Handle different prediction formats safely
    prediction_values = []
    if isinstance(predictions, dict):
        # Check for new structured format from train_model_for_stock
        if 'prediction' in predictions:
            avg_pred = predictions['prediction']
            print(f"[predict_single_ticker_chart] Using structured format prediction: {avg_pred}")
        else:
            # Old format: extract from individual model predictions
            for k, v in predictions.items():
                if isinstance(v, dict) and 'avg_pred' in v:
                    prediction_values.append(v['avg_pred'])
                elif isinstance(v, (int, float)):
                    prediction_values.append(float(v))
                elif isinstance(v, (list, np.ndarray)) and len(v) > 0:
                    prediction_values.append(float(v[-1]) if hasattr(v[-1], '__float__') else 0.0)
            
            avg_pred = np.mean(prediction_values) if len(prediction_values) > 0 else 0.0
            print(f"[predict_single_ticker_chart] Extracted individual predictions: {prediction_values}, average: {avg_pred}")
    else:
        avg_pred = float(predictions) if predictions is not None else 0.0
        print(f"[predict_single_ticker_chart] Single prediction value: {avg_pred}")
    
    # Ensure prediction is reasonable (should be small decimal like 0.0007 for 0.07%)
    if abs(avg_pred) > 1.0:
        print(f"[predict_single_ticker_chart] WARNING: Prediction {avg_pred} seems too large, capping at ±0.1")
        avg_pred = np.sign(avg_pred) * min(abs(avg_pred), 0.1)
    
    print(f"[predict_single_ticker_chart] Final prediction: {avg_pred} ({avg_pred*100:.3f}%)")
    
    # History - show actual price movement over last prediction_window days
    x_hist = np.arange(-prediction_window, 1)
    plt.plot(x_hist, hist_norm.values, color='cyan', linewidth=2, label="History")
    
    # Prediction (dashed, same color)
    pred_start = hist_norm.values[-1]  # Start from last historical price
    pred_end = pred_start * (1 + avg_pred)  # End price after prediction_window days
    
    print(f"[predict_single_ticker_chart] Chart: start={pred_start:.2f}, end={pred_end:.2f}")
    
    # Create linear interpolation from current price to predicted price
    pred_path = []
    for i in range(prediction_window + 1):
        # Linear interpolation: start + (end - start) * (i / prediction_window)
        progress = i / prediction_window if prediction_window > 0 else 0
        pred_value = pred_start + (pred_end - pred_start) * progress
        pred_path.append(pred_value)
    
    x_pred = np.arange(0, prediction_window+1)
    plt.plot(x_pred, pred_path, color='cyan', linewidth=2, linestyle='--', label="Prediction")
    plt.xlabel(f"Days (0 = present, -N = history, N = forecast)", color='white')
    plt.ylabel("Normalized Price (Start = 100)", color='white')
    plt.title(f"Single Ticker Prediction (window={prediction_window} days) - Pred: {avg_pred*100:.3f}%", color='white', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation showing the actual prediction percentage
    plt.text(0.02, 0.98, f"Prediction: {avg_pred*100:.3f}% over {prediction_window} days", 
             transform=plt.gca().transAxes, color='white', fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
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
        show_type = data.get('showType', 'top')
        is_custom = index == 'CUSTOM'
        if is_custom:
            ticker_to_analyze = custom_ticker
            if not ticker_to_analyze:
                return jsonify({'error': 'Please enter a valid ticker symbol (e.g., AAPL, MSFT)'}), 400
            
            print("Downloading market reference data...")
            market_data_cache = download_market_data_cache(start_date, force_refresh)
            spy_df = download_single_ticker_data('SPY', start_date)
            if spy_df is not None:
                market_condition, market_strength = detect_market_regime(spy_df)
            else:
                market_condition, market_strength = 'sideways', 0.5
            if model_selection == 'auto':
                selected_models = select_models_for_market(market_condition, is_custom)
                print(f"[predict] Single ticker auto model selection: {selected_models} for market condition: {market_condition}")
            elif not selected_models:
                selected_models = [2, 7, 6]
                print(f"[predict] Single ticker default model selection: {selected_models}")
            else:
                print(f"[predict] Single ticker user-selected models: {selected_models}")
            
            print(f"[predict] Final selected_models for single ticker: {selected_models}")
            df = download_single_ticker_data(ticker_to_analyze, start_date)
            if df is None or df.empty or len(df) < 50:
                return jsonify({'error': f'❌ Could not fetch data for {ticker_to_analyze}. Please check the ticker symbol and try again.'}), 400
            features_df = add_features_to_stock(ticker_to_analyze, df, prediction_window, market_data_cache)
            if features_df is None or features_df.empty:
                return jsonify({'error': '❌ Could not process technical indicators. Data may be corrupted or insufficient.'}), 400
            model_preds = train_model_single(features_df, selected_models, market_condition, market_strength)
            if not model_preds:
                return jsonify({'error': '❌ Could not train machine learning models. Insufficient data or technical issues.'}), 400
            # Use improved confidence interval calculation for single ticker
            ci = fixed_single_ticker_prediction(df, model_preds, market_condition, market_strength, confidence_interval, prediction_window)
            chart_image = predict_single_ticker_chart(features_df, model_preds, prediction_window)
            # Get last close price for the ticker
            last_close = float(df['Close'].iloc[-1]) if 'Close' in df.columns and not df.empty else None
            response = {
                'index_prediction': {
                    'ticker': ticker_to_analyze,
                    'index_name': ticker_to_analyze,
                    'pred': ci['prediction'],
                    'lower': ci['lower_bound'],
                    'upper': ci['upper_bound'],
                    'close': last_close
                },
                'selected_models': selected_models,
                'market_condition': market_condition,
                'market_strength': market_strength,
                'plot_image': chart_image,
                'system_messages': []
            }
            print(f"[predict] Single ticker response selected_models: {response['selected_models']}")
            return jsonify(sanitize_for_json(response))
        else:
            print(f"Starting multi-ticker analysis for {index} with {num_stocks} stocks...")
            print("Downloading market reference data...")
            market_data_cache = download_market_data_cache(start_date, force_refresh)
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
            processed_data = add_features_parallel(stock_data, prediction_window, market_data_cache)
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
                
                # Debug: print model predictions for first few stocks
                if len(stock_predictions) < 3:
                    print(f"[DEBUG] {ticker} model predictions: {model_predictions}")
                
                # Use improved confidence interval calculation for each stock
                stock_df = stock_data.get(ticker)
                ci = fixed_single_ticker_prediction(
                    stock_df,
                    model_predictions,
                    market_condition,
                    market_strength,
                    confidence_interval,
                    prediction_window
                )
                last_close = float(stock_df['Close'].iloc[-1]) if stock_df is not None and 'Close' in stock_df.columns and not stock_df.empty else None
                stock_predictions.append({
                    'ticker': ticker,
                    'pred': ci['prediction'],
                    'lower': ci['lower_bound'],
                    'upper': ci['upper_bound'],
                    'close': last_close
                })
            # Sort based on showType parameter (default to 'top' if not specified)
            show_type = data.get('showType', 'top')
            # Sort predictions:
            # For 'top': highest predictions first (most positive)
            # For 'worst': lowest predictions first (most negative)
            stock_predictions.sort(key=lambda x: x['pred'], reverse=(show_type == 'top'))
            stock_predictions = stock_predictions[:num_stocks]

            # Use the ETF's own prediction for the main index prediction
            etf_prediction_result = trained_models.get(etf_ticker)
            if etf_ticker and etf_prediction_result:
                etf_df = stock_data.get(etf_ticker)
                ci = fixed_single_ticker_prediction(
                    etf_df,
                    etf_prediction_result,
                    market_condition,
                    market_strength,
                    confidence_interval,
                    prediction_window
                )
                last_close = float(etf_df['Close'].iloc[-1]) if etf_df is not None and 'Close' in etf_df.columns and not etf_df.empty else None
                index_prediction = ci['prediction']
                index_lower = ci['lower_bound']
                index_upper = ci['upper_bound']
                index_name_for_response = etf_ticker
            else:
                # Fallback to averaging top 5 stocks if ETF prediction is not available
                index_prediction = np.mean([s['pred'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_lower = np.mean([s['lower'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_upper = np.mean([s['upper'] for s in stock_predictions[:5]]) if stock_predictions else 0
                index_name_for_response = index
                last_close = None

            # Store show_type temporarily for chart creation
            plt._show_type = show_type
            chart_image = create_multi_stock_prediction_chart(stock_data, stock_predictions, prediction_window)
            # Clean up temporary attribute
            if hasattr(plt, '_show_type'):
                del plt._show_type
                
            # Add rank based on show_type
            # For both cases, lower rank (1,2,3...) means "better" at what we're looking for
            # For 'top', rank 1 = highest positive prediction
            # For 'worst', rank 1 = lowest negative prediction
            for i, stock in enumerate(stock_predictions):
                rank = i + 1  # Simple 1-based index for both cases
                stock['rank'] = rank
                
            response = {
                'index_prediction': {
                    'ticker': index_name_for_response,
                    'index_name': index_name_for_response,
                    'pred': index_prediction,
                    'lower': index_lower,
                    'upper': index_upper,
                    'close': last_close
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
        return jsonify(sanitize_for_json(response))
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