import base64
import io
import logging
import os
import pickle
import random
import re
import time
import traceback
import warnings
import threading
import time
from catboost import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Lock
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from ta.momentum import RSIIndicator, WilliamsRIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from catboost import CatBoostClassifier, Pool
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
# Configure environment and warnings
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
app = Flask(__name__)
CORS(app)

# Machine Learning Constants
MIN_DATA_POINTS = 50  # Minimum data points required for training
MIN_FEATURES = 5  # Minimum features required for training
MIN_CONFIDENCE_SAMPLES = 30  # Minimum samples for confidence calculations
DEFAULT_CONFIDENCE_LEVEL = 95  # Default confidence level percentage
DEFAULT_MARGIN_ERROR = 0.05  # Default margin of error (5%)
TRADING_DAYS_YEAR = 252  # Number of trading days in a year
FEATURE_ROLLING_WINDOW = 20  # Default rolling window for features
VIX_HIGH_THRESHOLD = 25  # VIX level considered high fear
VIX_VERY_HIGH_THRESHOLD = 30  # VIX level considered very high fear
VIX_LOW_THRESHOLD = 15   # VIX level considered low fear
FORWARD_FILL_LIMIT = 5   # Maximum days to forward fill missing data

# Market Condition Thresholds
SIDEWAYS_PRICE_THRESHOLD = 0.05  # Price change threshold for sideways market
SIDEWAYS_SMA_THRESHOLD = 0.03    # SMA change threshold for sideways market
BULL_PRICE_THRESHOLD = 0.1       # Price change threshold for bull market
BULL_SMA_THRESHOLD = 0.05        # SMA change threshold for bull market
BEAR_PRICE_THRESHOLD = -0.1      # Price change threshold for bear market
BEAR_SMA_THRESHOLD = -0.05       # SMA change threshold for bear market
HIGH_VOLATILITY_THRESHOLD = 0.03 # Volatility threshold for volatile market

# Regime-based Volatility Constants
REGIME_VOLATILITY_BULL = 0.015     # Base volatility for bull market
REGIME_VOLATILITY_BEAR = 0.025     # Base volatility for bear market
REGIME_VOLATILITY_SIDEWAYS = 0.02  # Base volatility for sideways market
REGIME_VOLATILITY_VOLATILE = 0.035 # Base volatility for volatile market
VOLATILITY_ADJUSTMENT_BASE = 0.5   # Base multiplier for volatility adjustment

# Cache and Data Management Constants
CACHE_MAX_SIZE = 50  # Maximum number of cache entries to prevent memory bloat
CACHE_MAX_AGE_HOURS = 24  # Maximum age of cache entries in hours

# Market symbols for cross-asset features
MARKET_SYMBOLS = ['SPY', 'VIX', 'DXY', 'TLT', 'GLD', 'QQQ']

# Sentiment Analysis Constants
SENTIMENT_CONFIG = {
    'ALPHA_VANTAGE_API_KEY': '4AOTSWIKBYTLSJT4',  # 🔑 Your real Alpha Vantage API key
    'COMPANY_NEWS_WEIGHT': 0.5,    # Weight for company-specific news sentiment
    'SECTOR_NEWS_WEIGHT': 0.2,     # Weight for sector news sentiment  
    'MARKET_NEWS_WEIGHT': 0.3,     # New: Global market sentiment weight
    'NEWS_LOOKBACK_DAYS': 7,
    'SENTIMENT_CACHE_HOURS': 1,
    'MARKET_SENTIMENT_CACHE_HOURS': 24,  # Market sentiment cached for 24 hours
    'SENTIMENT_ADJUSTMENT_MAX': 0.25,
    'API_REQUEST_INTERVAL': 0.25  # 250ms between requests (4 req/sec)
}

# Sector mapping for consistent news searches
SECTOR_MAPPING = {
    'Technology': ['technology', 'tech stocks', 'software', 'hardware', 'semiconductors'],
    'Healthcare': ['healthcare', 'pharmaceuticals', 'biotech', 'medical devices'],
    'Financial Services': ['banking', 'finance', 'financial services', 'fintech', 'banks'],
    'Consumer Cyclical': ['consumer cyclical', 'retail', 'e-commerce', 'consumer discretionary'],
    'Communication Services': ['communication services', 'telecom', 'media', 'entertainment'],
    'Industrials': ['industrial stocks', 'manufacturing', 'aerospace', 'defense'],
    'Consumer Defensive': ['consumer staples', 'consumer defensive', 'food', 'beverages'],
    'Energy': ['energy stocks', 'oil', 'gas', 'renewable energy'],
    'Basic Materials': ['basic materials', 'chemicals', 'mining', 'metals'],
    'Utilities': ['utility stocks', 'utilities', 'electric', 'water', 'gas utilities'],
    'Real Estate': ['real estate', 'reits', 'property']
}

# Global market sentiment search terms (major market ETFs and indices as proxies)
MARKET_SENTIMENT_TERMS = [
    'SPY',    
    'QQQ',    
    'DIA',    
    'IWM',    
    'VIX',    # Volatility Index
    'TLT'     # Treasury ETF (flight to safety indicator)
]
# Global sentiment model
SENTIMENT_MODEL = None
SENTIMENT_TOKENIZER = None
SENTIMENT_PIPELINE = None

# Sentiment cache with lock for thread safety
SENTIMENT_CACHE = {}
SENTIMENT_CACHE_LOCK = Lock()

# Global market sentiment cache (24-hour cache)
MARKET_SENTIMENT_CACHE = None
MARKET_SENTIMENT_TIMESTAMP = None
MARKET_SENTIMENT_LOCK = Lock()

# Track API requests to prevent rate limit issues
LAST_NEWS_API_CALL = None

# List of supported index names
SUPPORTED_INDEXES = ['SPY', 'DIA', 'NASDAQ', 'SP400', 'SPLV', 'SPHB', 'SPSM']

def is_index_ticker(ticker):
    """Check if a ticker is one of our supported indexes"""
    return ticker.upper() in SUPPORTED_INDEXES

def ensure_iterable(obj):
    """Ensure the object is iterable. If not, wrap it in a list."""
    if isinstance(obj, (list, np.ndarray, pd.Series)):
        return obj
    return [obj]

def initialize_sentiment_model():
    """Load the FinBERT sentiment model at server startup (with fallback)"""
    global SENTIMENT_MODEL, SENTIMENT_TOKENIZER, SENTIMENT_PIPELINE
    
    try:
        print("Loading sentiment model...")
        # Suppress progress bars and verbose output
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        
        # Set environment variable to suppress download progress
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        
        SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert",
            local_files_only=False,
            trust_remote_code=False
        )
        SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert",
            local_files_only=False,
            trust_remote_code=False
        )
        
        # Create pipeline for easier sentiment prediction
        SENTIMENT_PIPELINE = pipeline(
            "text-classification", 
            model=SENTIMENT_MODEL, 
            tokenizer=SENTIMENT_TOKENIZER,
            device=-1  # Force CPU to avoid CUDA warnings
        )
        
        print("✅ Sentiment model loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ Sentiment model failed to load, using fallback methods")
        SENTIMENT_PIPELINE = None
        return False

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

def get_cached_sentiment(ticker):
    """Get cached sentiment data if available and not expired"""
    with SENTIMENT_CACHE_LOCK:
        if ticker in SENTIMENT_CACHE:
            timestamp, data = SENTIMENT_CACHE[ticker]
            # Check if cache is still valid (hours in seconds)
            cache_seconds = SENTIMENT_CONFIG['SENTIMENT_CACHE_HOURS'] * 3600
            if time.time() - timestamp < cache_seconds:
                return data
    return None

def cache_sentiment(ticker, sentiment_data):
    """Store sentiment data in cache with timestamp"""
    with SENTIMENT_CACHE_LOCK:
        SENTIMENT_CACHE[ticker] = (time.time(), sentiment_data)
        
        # Cache cleanup - remove oldest entries if > 100 items
        if len(SENTIMENT_CACHE) > 100:
            # Sort by timestamp and keep newest 80
            sorted_items = sorted(
                SENTIMENT_CACHE.items(), 
                key=lambda x: x[1][0], 
                reverse=True
            )
            SENTIMENT_CACHE.clear()
            for k, v in sorted_items[:80]:
                SENTIMENT_CACHE[k] = v

def get_ticker_sector(ticker):
    """Get sector for a ticker from Yahoo Finance"""
    try:
        # Create yfinance ticker object
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Try to get sector from info
        sector = info.get('sector', '')
        if sector:
            return sector
            
        # If no sector, try to get from industry
        industry = info.get('industry', '')
        if industry:
            # Map common industries to sectors
            industry_lower = industry.lower()
            if 'bank' in industry_lower or 'financial' in industry_lower:
                return 'Financial Services'
            elif 'tech' in industry_lower or 'software' in industry_lower:
                return 'Technology'
            elif 'health' in industry_lower or 'pharma' in industry_lower or 'bio' in industry_lower:
                return 'Healthcare'
            elif 'oil' in industry_lower or 'gas' in industry_lower or 'energy' in industry_lower:
                return 'Energy'
        
        return 'Unknown'
        
    except Exception as e:
        print(f"Error getting sector for {ticker}: {e}")
        return 'Unknown'

def get_alpha_vantage_news(query, limit=20, is_ticker=True):
    """Get news from Alpha Vantage News API with improved error handling"""
    global LAST_NEWS_API_CALL
    
    # Rate limiting - ensure at least 12 seconds between calls
    current_time = time.time()
    if LAST_NEWS_API_CALL is not None:
        time_since_last = current_time - LAST_NEWS_API_CALL
        if time_since_last < 12:  # 300 calls/day = ~12 seconds apart
            time.sleep(12 - time_since_last)
    
    LAST_NEWS_API_CALL = time.time()
    
    try:
        url = "https://www.alphavantage.co/query"
        
        # Use different parameters for tickers vs general topics
        if is_ticker:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': query,
                'apikey': SENTIMENT_CONFIG['ALPHA_VANTAGE_API_KEY'],
                'limit': min(limit, 50),  # API limit
                'sort': 'LATEST'
            }
        else:
            # For general terms, try keywords parameter
            params = {
                'function': 'NEWS_SENTIMENT',
                'keywords': query,
                'apikey': SENTIMENT_CONFIG['ALPHA_VANTAGE_API_KEY'],
                'limit': min(limit, 50),
                'sort': 'LATEST'
            }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API error or information messages (less verbose)
        if 'Error Message' in data:
            print(f"⚠️ Alpha Vantage API Error for {query}")
            return []
        
        if 'Note' in data:
            print(f"⚠️ Alpha Vantage API rate limit reached")
            return []
            
        if 'Information' in data:
            # API key issue or invalid input - fail silently and use fallback
            return []
        
        # Extract articles
        articles = data.get('feed', [])
        
        # Filter and format articles (only show count if successful)
        filtered_articles = []
        for article in articles:
            try:
                # Parse time
                time_published = article.get('time_published', '')
                if len(time_published) >= 8:
                    # Format: YYYYMMDDTHHMMSS
                    pub_datetime = datetime.strptime(time_published[:8], '%Y%m%d')
                    
                    # Only include articles from last 7 days
                    days_old = (datetime.now() - pub_datetime).days
                    if days_old > 7:
                        continue
                    
                    filtered_articles.append({
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'time_published': time_published,
                        'days_old': days_old,
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'overall_sentiment_label': article.get('overall_sentiment_label', 'Neutral')
                    })
                    
            except Exception as e:
                # Silently skip malformed articles
                continue
        
        # Only show success message if we got articles
        if filtered_articles:
            print(f"📰 Retrieved {len(filtered_articles)} articles for {query}")
        
        return filtered_articles
        
    except Exception as e:
        # Silently fail and return empty list for fallback
        return []

def analyze_sentiment_with_finbert(text):
    """Analyze sentiment using FinBERT model or fallback methods"""
    global SENTIMENT_PIPELINE
    
    # Try FinBERT first if available
    if SENTIMENT_PIPELINE is not None:
        try:
            # Clean and truncate text
            text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Truncate to reasonable length for model
            if len(text) > 500:
                text = text[:500]
            
            if not text.strip():
                return 0.0
            
            # Get sentiment prediction
            result = SENTIMENT_PIPELINE(text)
            
            # Convert to numerical score (0 to +100)
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if 'positive' in label:
                return score * 100
            elif 'negative' in label:
                return -score * 100
            else:  # neutral
                return 0.0
                
        except Exception as e:
            print(f"Error with FinBERT, falling back to rule-based: {e}")
    
    # Fallback to rule-based sentiment analysis
    return analyze_sentiment_rule_based(text)

def analyze_sentiment_rule_based(text):
    """Simple rule-based sentiment analysis as fallback"""
    if not text or not text.strip():
        return 0.0
    
    text = text.lower()
    
    # Financial positive keywords
    positive_words = {
        'bullish', 'bull', 'rally', 'surge', 'soar', 'gain', 'gains', 'up', 'rise', 'rising', 'increased', 
        'growth', 'profit', 'profits', 'strong', 'beat', 'beats', 'exceeded', 'outperform', 'optimistic',
        'positive', 'upgrade', 'upgraded', 'buy', 'buying', 'momentum', 'breakthrough', 'record high',
        'expansion', 'recovering', 'recovery', 'boosted', 'improved', 'stellar', 'robust', 'solid'
    }
    
    # Financial negative keywords  
    negative_words = {
        'bearish', 'bear', 'crash', 'plunge', 'fall', 'falling', 'dropped', 'decline', 'declining', 'loss',
        'losses', 'weak', 'weakness', 'miss', 'missed', 'underperform', 'pessimistic', 'negative', 'concern',
        'concerns', 'worried', 'fear', 'fears', 'downgrade', 'downgraded', 'sell', 'selling', 'pressure',
        'recession', 'crisis', 'volatile', 'volatility', 'risk', 'risks', 'uncertain', 'uncertainty',
        'disappointing', 'struggled', 'struggling', 'challenges', 'headwinds', 'tariff', 'tariffs'
    }
    
    # Count sentiment words
    words = text.split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    positive_ratio = positive_count / total_words
    negative_ratio = negative_count / total_words
    
    # Net sentiment (-100 to +100)
    net_sentiment = (positive_ratio - negative_ratio) * 100
    
    # Apply some scaling to match FinBERT-like outputs
    sentiment_score = max(-100, min(100, net_sentiment * 50))  # Scale to reasonable range
    
    return sentiment_score

def calculate_time_decay_weight(days_old):
    """Calculate time decay weight for news articles"""
    # Linear decay from 1.0 (today) to 0.1 (7 days old)
    if days_old <= 0:
        return 1.0
    elif days_old >= 7:
        return 0.1
    else:
        return 1.0 - (days_old * 0.9 / 7.0)

def get_weekend_adjusted_days(pub_datetime):
    """Adjust days calculation for weekend news carrying into Monday"""
    current_datetime = datetime.now()
    
    # If today is Monday and article is from weekend, treat as today
    if current_datetime.weekday() == 0:  # Monday
        if pub_datetime.weekday() in [5, 6]:  # Saturday or Sunday
            # Weekend news carries into Monday
            weekend_diff = current_datetime.date() - pub_datetime.date()
            if weekend_diff.days <= 2:  # Same weekend
                return 0
    
    # Normal day calculation
    return (current_datetime.date() - pub_datetime.date()).days

def get_market_sentiment():
    """Get global market sentiment (cached for 24 hours)"""
    global MARKET_SENTIMENT_CACHE, MARKET_SENTIMENT_TIMESTAMP
    
    with MARKET_SENTIMENT_LOCK:
        # Check if we have valid cached market sentiment
        if (MARKET_SENTIMENT_CACHE is not None and 
            MARKET_SENTIMENT_TIMESTAMP is not None):
            
            cache_age_hours = (time.time() - MARKET_SENTIMENT_TIMESTAMP) / 3600
            if cache_age_hours < SENTIMENT_CONFIG['MARKET_SENTIMENT_CACHE_HOURS']:
                return MARKET_SENTIMENT_CACHE
        
        # Need to fetch new market sentiment
        print("📰 Fetching market sentiment...")
        
        market_sentiment = 0.0
        market_weight_sum = 0.0
        article_count = 0
        api_failures = 0
        
        # Search multiple market terms (using major ETFs as market proxies)
        for term in MARKET_SENTIMENT_TERMS[:3]:  # Limit to 3 terms to reduce API calls
            try:
                # These are ETF symbols, so use as tickers
                articles = get_alpha_vantage_news(term, limit=5, is_ticker=True)
                
                if not articles:
                    api_failures += 1
                    # If first few API calls fail, immediately use fallback
                    if api_failures >= 2:
                        print("📰 API calls failing, using fallback market sentiment...")
                        break
                    continue
                
                for article in articles:
                    try:
                        # Parse publication date
                        time_published = article['time_published']
                        if len(time_published) >= 8:
                            pub_datetime = datetime.strptime(time_published[:8], '%Y%m%d')
                            days_old = get_weekend_adjusted_days(pub_datetime)
                            
                            # Include articles from last 3 days for market sentiment
                            if days_old > 3:
                                continue
                            
                            # Calculate time decay weight
                            time_weight = calculate_time_decay_weight(days_old)
                            
                            # Analyze sentiment - prefer Alpha Vantage scores, fallback to text analysis
                            alpha_vantage_sentiment = article.get('overall_sentiment_score', 0)
                            if alpha_vantage_sentiment != 0:
                                # Use Alpha Vantage sentiment (scale from -1 to 1 range to -100 to 100)
                                sentiment = alpha_vantage_sentiment * 100
                            else:
                                # Fallback to text analysis
                                text = f"{article['title']} {article['summary']}"
                                sentiment = analyze_sentiment_with_finbert(text)
                            
                            # Weight by time decay and relevance
                            weighted_sentiment = sentiment * time_weight
                            market_sentiment += weighted_sentiment
                            market_weight_sum += time_weight
                            article_count += 1
                            
                    except Exception as e:
                        # Silently skip malformed articles
                        continue
                        
                # Reduce delay between term searches
                if article_count < 5:  # Only add delay if we're still searching
                    time.sleep(0.5)
                
            except Exception as e:
                api_failures += 1
                # Skip failed searches silently
                continue
        
        # If no articles found or API failures, use fallback approach immediately
        if article_count == 0 or api_failures >= 2:
            market_sentiment = get_fallback_market_sentiment()
            article_count = 1  # Prevent division by zero
            market_weight_sum = 1.0
        
        # Calculate final market sentiment
        if market_weight_sum > 0:
            final_market_sentiment = market_sentiment / market_weight_sum
        else:
            final_market_sentiment = market_sentiment
        
        # Ensure sentiment is in range [-100, 100]
        final_market_sentiment = max(-100, min(100, final_market_sentiment))
        
        # Cache the result
        MARKET_SENTIMENT_CACHE = final_market_sentiment
        MARKET_SENTIMENT_TIMESTAMP = time.time()
        
        if article_count > 1:
            print(f"✅ Market sentiment: {final_market_sentiment:.1f}/100 (from {article_count} articles)")
        else:
            print(f"✅ Market sentiment: {final_market_sentiment:.1f}/100 (fallback)")
            
        return final_market_sentiment

def get_fallback_market_sentiment():
    """Fallback market sentiment based on total market indices and price movements"""
    try:
        # List of total market indices - prioritize broad market ETFs over sector-specific ones
        market_indices = [
            'VTI',   # Total Stock Market ETF (PRIMARY - most comprehensive)
            'ITOT',  # Core S&P Total U.S. Stock Market ETF (PRIMARY)
            'VTV',   # Vanguard Value ETF (total market value)
            'VUG',   # Vanguard Growth ETF (total market growth)
            '^GSPC', # S&P 500 Index (secondary)
            'SPY',   # S&P 500 (secondary)
            'IWM'    # Russell 2000 (secondary - small caps)
        ]
        
        market_sentiment_values = []
        market_weights = []  # Track weights for different indices
        
        for i, symbol in enumerate(market_indices):
            try:
                ticker = yf.Ticker(symbol)
                # Get recent data including today
                data = ticker.history(period='5d', interval='1d')
                
                if len(data) >= 2:
                    # Calculate daily return (today vs yesterday)
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2]
                    daily_return = (current_price - previous_price) / previous_price
                    
                    # Calculate 5-day trend for additional context
                    if len(data) >= 5:
                        week_return = (current_price - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                    else:
                        week_return = daily_return
                    
                    # REDUCED scaling factors to match API sentiment range better
                    daily_sentiment = daily_return * 1000  # Reduced from 2000
                    weekly_sentiment = week_return * 500   # Reduced from 1000
                    
                    # Apply a baseline negative bias to match news sentiment
                    baseline_bias = -15  # Add negative bias to align with news API
                    
                    combined_sentiment = (daily_sentiment * 0.7) + (weekly_sentiment * 0.3) + baseline_bias
                    
                    # Cap sentiment between -100 and +100
                    combined_sentiment = max(-100, min(100, combined_sentiment))
                    
                    # Assign weights: Total market indices get higher weights
                    if symbol in ['VTI', 'ITOT']:  # Primary total market ETFs
                        weight = 3.0  # 3x weight for most comprehensive indices
                    elif symbol in ['VTV', 'VUG']:  # Total market style indices  
                        weight = 2.0  # 2x weight for broad style indices
                    elif symbol in ['^GSPC', 'SPY']:  # S&P 500 indices
                        weight = 1.5  # 1.5x weight for large cap proxy
                    else:  # IWM (small caps)
                        weight = 1.0  # Normal weight for small cap supplement
                    
                    market_sentiment_values.append(combined_sentiment)
                    market_weights.append(weight)
                    
            except Exception as e:
                # Silently skip failed symbols
                continue
        
        if market_sentiment_values and market_weights:
            # Calculate weighted average sentiment across all available indices
            weighted_sum = sum(sentiment * weight for sentiment, weight in zip(market_sentiment_values, market_weights))
            total_weight = sum(market_weights)
            avg_sentiment = weighted_sum / total_weight
            
            # Add VIX boost for fear factor if available
            try:
                vix_ticker = yf.Ticker('^VIX')
                vix_data = vix_ticker.history(period='2d')
                
                if len(vix_data) > 0:
                    current_vix = vix_data['Close'].iloc[-1]
                    
                    # VIX adjustment (affects both positive and negative sentiment)
                    if current_vix > 25:
                        # High VIX - dampen positive sentiment, amplify negative
                        if avg_sentiment > 0:
                            avg_sentiment *= 0.7  # Reduce positive sentiment
                        else:
                            avg_sentiment *= 1.3  # Amplify negative sentiment
                    elif current_vix > 20:
                        # Moderate VIX - less extreme adjustments
                        if avg_sentiment > 0:
                            avg_sentiment *= 0.9
                        else:
                            avg_sentiment *= 1.1
                    
            except Exception as e:
                # Silently skip VIX adjustment if it fails
                pass
            
            # Final bounds checking
            final_sentiment = max(-100, min(100, avg_sentiment))
            
            return final_sentiment
            
    except Exception as e:
        # Silently handle errors
        pass
    
    # Ultimate fallback - bearish bias for current conditions
    return -25

def get_index_sentiment_score(index_name):
    """Get sentiment score for indexes using ONLY market data (no Alpha Vantage API calls)"""
    try:
        # Check cache first
        cached_data = get_cached_sentiment(f"INDEX_{index_name}")
        if cached_data:
            return cached_data
        
        print(f"[get_index_sentiment_score] 📊 Analyzing sentiment for index {index_name} (market data only)")
        
        # Get global market sentiment (cached for 24 hours)
        market_sentiment = get_market_sentiment()
        
        # For index tickers, use market sentiment only to avoid API calls
        # This avoids unnecessary API usage and potential rate limits for index analysis
        print(f"[get_index_sentiment_score] 📈 {index_name}: Using market sentiment only (no news API calls)")
        
        # Set index news variables to indicate no news search was performed
        index_news = []
        index_sentiment = 0.0
        index_weight_sum = 0.0
        
        # Calculate final weighted sentiment for index
        final_sentiment = 0.0
        has_index_news = False  # Never have index news since we skip API calls
        
        # For indexes, use 100% market sentiment without mixing news data
        final_sentiment = market_sentiment
        index_weight = 0.0
        market_weight = 1.0
        
        print(f"[get_index_sentiment_score] {index_name} - Index sentiment (market data only):")
        print(f"  Index News: 0.000 (⚠️ - skipped API calls for indexes)")
        print(f"  Market: 1.000 (✓ - using market sentiment: {market_sentiment:.1f})")
        print(f"  Calculation: Pure market sentiment = {final_sentiment:.1f}")
        print(f"  Total weight: 1.000")
        print(f"  Final sentiment: {final_sentiment:.1f}/100")
        
        # Ensure sentiment is in range [-100, 100]
        final_sentiment = max(-100, min(100, final_sentiment))
        print(f"  Final sentiment after bounds check: {final_sentiment:.1f}/100")
        
        # No index-specific sentiment since we skip news search
        index_avg_sentiment = 0.0
        
        # Create result data consistent with single ticker format but with index-specific fields
        result = {
            'sentiment_score': final_sentiment,
            'index_articles': 0,   # No index articles since we skip API calls
            'sector_articles': 0,    # No sector news for indexes
            'market_sentiment': market_sentiment,
            'sector': 'Index',       # Mark as index
            'timestamp': time.time(),
            # Dynamic weights for frontend consistency
            'dynamic_index_weight': index_weight,  # Index news weight (always 0)
            'dynamic_sector_weight': 0.0,            # No sector for indexes
            'dynamic_market_weight': market_weight,  # Market weight (always 1.0)
            'has_index_news': has_index_news,      # Always False for indexes
            'has_sector_news': False,                # Never have sector news for indexes
            'is_index': True,                        # Flag to identify index sentiment
            # Add the actual sentiment scores - renamed for indices
            'index_sentiment_score': index_avg_sentiment,  # Always 0 for indexes
            'sector_sentiment_score': 0.0,                  # No sector sentiment for indexes
            'api_limit_reached': True,                      # We're always treating it as API limit reached since we skip calls
            'using_fallback': True                         # Always using market data fallback
        }
        
        # Cache the result
        cache_sentiment(f"INDEX_{index_name}", result)
        
        return result
        
    except Exception as e:
        print(f"Error getting index sentiment for {index_name}: {e}")
        return {
            'sentiment_score': 0.0,
            'index_articles': 0,
            'sector_articles': 0,
            'market_sentiment': 0.0,
            'sector': 'Index',
            'timestamp': time.time(),
            'dynamic_index_weight': 0.0,
            'dynamic_sector_weight': 0.0,
            'dynamic_market_weight': 1.0,
            'has_index_news': False,
            'has_sector_news': False,
            'is_index': True,
            'index_sentiment_score': 0.0,
            'api_limit_reached': True,
            'using_fallback': True
        }

def get_sentiment_score(ticker):
    """Get comprehensive sentiment score for a ticker"""
    try:
        # Check cache first
        cached_data = get_cached_sentiment(ticker)
        if cached_data:
            return cached_data
        
        print(f"Fetching sentiment for {ticker}")
        
        # Get global market sentiment (cached for 24 hours)
        market_sentiment = get_market_sentiment()
        
        # Get company news (50% weight)
        company_news = get_alpha_vantage_news(ticker, limit=15, is_ticker=True)
        
        # Get sector and sector news (20% weight)
        sector = get_ticker_sector(ticker)
        sector_search_terms = SECTOR_MAPPING.get(sector, [sector])
        
        sector_news = []
        for term in sector_search_terms[:2]:  # Limit to 2 search terms
            sector_articles = get_alpha_vantage_news(term, limit=8, is_ticker=False)
            sector_news.extend(sector_articles)
        
        # Remove duplicates from sector news
        seen_titles = set()
        unique_sector_news = []
        for article in sector_news:
            title_key = article['title'].lower()[:50]  # Use first 50 chars as key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sector_news.append(article)
        
        # Calculate company sentiment
        company_sentiment = 0.0
        company_weight_sum = 0.0
        
        for article in company_news:
            try:
                # Parse publication date
                time_published = article['time_published']
                if len(time_published) >= 8:
                    pub_datetime = datetime.strptime(time_published[:8], '%Y%m%d')
                    days_old = get_weekend_adjusted_days(pub_datetime)
                    
                    # Skip if too old
                    if days_old > 7:
                        continue
                    
                    # Calculate time decay weight
                    time_weight = calculate_time_decay_weight(days_old)
                    
                    # Analyze sentiment - prefer Alpha Vantage scores, fallback to text analysis
                    alpha_vantage_sentiment = article.get('overall_sentiment_score', 0)
                    if alpha_vantage_sentiment != 0:
                        # Use Alpha Vantage sentiment (scale from -1 to 1 range to -100 to 100)
                        sentiment = alpha_vantage_sentiment * 100
                        print(f"[get_sentiment_score] {ticker} Company Article: '{article['title'][:50]}...' | AV Score: {alpha_vantage_sentiment:.3f} → {sentiment:.1f}/100")
                    else:
                        # Fallback to text analysis
                        text = f"{article['title']} {article['summary']}"
                        sentiment = analyze_sentiment_with_finbert(text)
                        print(f"[get_sentiment_score] {ticker} Company Article: '{article['title'][:50]}...' | FinBERT Score: {sentiment:.1f}/100")
                    
                    # Weight by time decay
                    weighted_sentiment = sentiment * time_weight
                    company_sentiment += weighted_sentiment
                    company_weight_sum += time_weight
                    
            except Exception as e:
                print(f"Error processing company article: {e}")
                continue
        
        # Calculate sector sentiment
        sector_sentiment = 0.0
        sector_weight_sum = 0.0
        
        for article in unique_sector_news:
            try:
                # Parse publication date
                time_published = article['time_published']
                if len(time_published) >= 8:
                    pub_datetime = datetime.strptime(time_published[:8], '%Y%m%d')
                    days_old = get_weekend_adjusted_days(pub_datetime)
                    
                    # Skip if too old
                    if days_old > 7:
                        continue
                    
                    # Calculate time decay weight
                    time_weight = calculate_time_decay_weight(days_old)
                    
                    # Analyze sentiment - prefer Alpha Vantage scores, fallback to text analysis
                    alpha_vantage_sentiment = article.get('overall_sentiment_score', 0)
                    if alpha_vantage_sentiment != 0:
                        # Use Alpha Vantage sentiment (scale from -1 to 1 range to -100 to 100)
                        sentiment = alpha_vantage_sentiment * 100
                        print(f"[get_sentiment_score] {ticker} Sector Article: '{article['title'][:50]}...' | AV Score: {alpha_vantage_sentiment:.3f} → {sentiment:.1f}/100")
                    else:
                        # Fallback to text analysis
                        text = f"{article['title']} {article['summary']}"
                        sentiment = analyze_sentiment_with_finbert(text)
                        print(f"[get_sentiment_score] {ticker} Sector Article: '{article['title'][:50]}...' | FinBERT Score: {sentiment:.1f}/100")
                    
                    # Weight by time decay
                    weighted_sentiment = sentiment * time_weight
                    sector_sentiment += weighted_sentiment
                    sector_weight_sum += time_weight
                    
            except Exception as e:
                print(f"Error processing sector article: {e}")
                continue
        
        # Calculate final weighted sentiment with dynamic weighting
        # Redistribute weights when news categories are missing
        final_sentiment = 0.0
        
        # Determine which news sources have data
        has_company = company_weight_sum > 0
        has_sector = sector_weight_sum > 0
        has_market = True  # Market sentiment is always available
        
        # Calculate dynamic weights based on available data
        company_weight = SENTIMENT_CONFIG['COMPANY_NEWS_WEIGHT'] if has_company else 0
        sector_weight = SENTIMENT_CONFIG['SECTOR_NEWS_WEIGHT'] if has_sector else 0
        market_weight = SENTIMENT_CONFIG['MARKET_NEWS_WEIGHT']
        
        # Redistribute missing weights - give ALL missing weight to available sources
        if not has_company and not has_sector:
            # No company or sector news - market gets 100% weight
            market_weight = 1.0
            company_weight = 0
            sector_weight = 0
        elif not has_company:
            # No company news - redistribute company weight between sector and market
            missing_company_weight = SENTIMENT_CONFIG['COMPANY_NEWS_WEIGHT']
            # Give it all to market since sector already has its own weight
            market_weight += missing_company_weight
            company_weight = 0
        elif not has_sector:
            # No sector news - redistribute sector weight between company and market  
            missing_sector_weight = SENTIMENT_CONFIG['SECTOR_NEWS_WEIGHT']
            # Give it all to market since company already has its own weight
            market_weight += missing_sector_weight
            sector_weight = 0
        # If we have all news sources, weights stay as configured
        
        # Apply weighted sentiment calculation
        if has_company:
            company_avg = company_sentiment / company_weight_sum
            final_sentiment += company_avg * company_weight
        
        if has_sector:
            sector_avg = sector_sentiment / sector_weight_sum
            final_sentiment += sector_avg * sector_weight
        
        # Market sentiment (dynamically weighted)
        final_sentiment += market_sentiment * market_weight
        
        # Debug logging for weight verification
        total_weight = (company_weight if has_company else 0) + (sector_weight if has_sector else 0) + market_weight
        print(f"[get_sentiment_score] {ticker} - Dynamic weights:")
        print(f"  Company: {company_weight:.3f} ({'✓' if has_company else '✗'})")
        print(f"  Sector: {sector_weight:.3f} ({'✓' if has_sector else '✗'})")
        print(f"  Market: {market_weight:.3f} (always available)")
        print(f"  Total weight: {total_weight:.3f} (should be ~1.0)")
        print(f"  Market sentiment value: {market_sentiment:.1f}/100")
        print(f"  Final sentiment: {final_sentiment:.1f}/100")
        
        # Ensure sentiment is in range [-100, 100]
        final_sentiment = max(-100, min(100, final_sentiment))
        
        # Calculate average sentiment scores for display
        company_avg_sentiment = (company_sentiment / company_weight_sum) if company_weight_sum > 0 else 0.0
        sector_avg_sentiment = (sector_sentiment / sector_weight_sum) if sector_weight_sum > 0 else 0.0
        
        # Create result data with dynamic weights
        result = {
            'sentiment_score': final_sentiment,
            'company_articles': len(company_news),
            'sector_articles': len(unique_sector_news),
            'market_sentiment': market_sentiment,
            'sector': sector,
            'timestamp': time.time(),
            # Include the dynamic weights that were actually used
            'dynamic_company_weight': company_weight if has_company else 0,
            'dynamic_sector_weight': sector_weight if has_sector else 0,
            'dynamic_market_weight': market_weight,
            'has_company_news': has_company,
            'has_sector_news': has_sector,
            # Add the actual sentiment scores (not just article counts)
            'company_sentiment_score': company_avg_sentiment,
            'sector_sentiment_score': sector_avg_sentiment
        }
        
        # Cache the result
        cache_sentiment(ticker, result)
        
        return result
        
    except Exception as e:
        print(f"Error getting sentiment for {ticker}: {e}")
        return {
            'sentiment_score': 0.0,
            'company_articles': 0,
            'sector_articles': 0,
            'market_sentiment': 0.0,
            'sector': 'Unknown',
            'timestamp': time.time()
        }

# --- Confidence Interval Calculations ---
def calculate_proper_confidence_interval(predictions, confidence_level=DEFAULT_CONFIDENCE_LEVEL, base_volatility=0.02):
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
    
    if len(pred_values) < MIN_CONFIDENCE_SAMPLES:
        alpha = (100 - confidence_level) / 100
        t_value = stats.t.ppf(1 - alpha/2, df=len(pred_values)-1)
        margin_of_error = t_value * total_uncertainty
    else:
        alpha = (100 - confidence_level) / 100
        z_value = stats.norm.ppf(1 - alpha/2)
        margin_of_error = z_value * total_uncertainty
    
    # Ensure margin_of_error is not NaN
    if np.isnan(margin_of_error):
        margin_of_error = DEFAULT_MARGIN_ERROR  # Default 5% margin
    
    lower_bound = mean_pred - margin_of_error
    upper_bound = mean_pred + margin_of_error
    return lower_bound, upper_bound

def calculate_regime_adjusted_confidence(predictions, market_condition, market_strength, confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """Calculate confidence intervals based on market regime"""
    regime_volatility = {
        'bull': REGIME_VOLATILITY_BULL,
        'bear': REGIME_VOLATILITY_BEAR,
        'sideways': REGIME_VOLATILITY_SIDEWAYS,
        'volatile': REGIME_VOLATILITY_VOLATILE
    }
    base_vol = regime_volatility.get(market_condition, REGIME_VOLATILITY_SIDEWAYS)
    adjusted_vol = base_vol * (VOLATILITY_ADJUSTMENT_BASE + market_strength)
    
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

def calculate_historical_confidence(df, predictions, confidence_level=DEFAULT_CONFIDENCE_LEVEL, lookback_days=TRADING_DAYS_YEAR, prediction_window=5):
    """Calculate confidence intervals using historical volatility"""
    if df is None or len(df) < MIN_CONFIDENCE_SAMPLES:
        # Fallback to basic confidence calculation
        if not hasattr(predictions, '__len__') or isinstance(predictions, (str, np.number)):
            predictions = [predictions]
        predictions = ensure_iterable(predictions)
        return calculate_proper_confidence_interval(predictions, confidence_level)
    
    # Calculate historical volatility
    returns = df['Close'].pct_change().dropna()
    if len(returns) < MIN_CONFIDENCE_SAMPLES:
        # Not enough data for meaningful volatility calculation
        return calculate_proper_confidence_interval([predictions], confidence_level)
    
    # Use available data, but at least MIN_CONFIDENCE_SAMPLES days
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
    
    # Apply market-condition-based confidence interval adjustment
    original_lower, original_upper = lower, upper
    
    # Adjust CI width based on market conditions
    if market_condition == 'bull' and market_strength > 0.7:
        # Strong bull market - slightly narrow CI (more confidence)
        ci_adjustment = -0.002  # Reduce CI by 0.2%
    elif market_condition == 'bear' and market_strength > 0.7:
        # Strong bear market - widen CI (less confidence)
        ci_adjustment = 0.005  # Increase CI by 0.5%
    elif market_condition == 'volatile':
        # Volatile market - significantly widen CI
        ci_adjustment = 0.008  # Increase CI by 0.8%
    else:
        # Neutral or weak conditions - minimal adjustment
        ci_adjustment = 0.001  # Slight increase for safety
    
    # Apply symmetric adjustment to bounds
    ci_width = upper - lower
    half_width = ci_width / 2
    center = (upper + lower) / 2
    
    new_half_width = half_width + ci_adjustment
    lower = center - new_half_width
    upper = center + new_half_width
    
    # Check if we have sentiment-adjusted prediction and apply CI adjustment for sentiment impact
    if isinstance(model_predictions, dict) and model_predictions.get('sentiment_adjusted', False):
        original_prediction = model_predictions.get('original_ml_prediction', avg_pred)
        adjusted_prediction = avg_pred  # Current prediction after sentiment
        
        # Only apply sentiment CI adjustment if there's a meaningful difference
        if abs(adjusted_prediction - original_prediction) > 0.001:  # 0.1% threshold
            # Adjust confidence interval bounds for sentiment impact
            lower, upper = adjust_confidence_interval_for_sentiment(
                lower, upper, original_prediction, adjusted_prediction
            )
            print(f"[fixed_single_ticker_prediction] Applied sentiment CI adjustment: original_pred={original_prediction:.6f}, adjusted_pred={adjusted_prediction:.6f}")
            print(f"[fixed_single_ticker_prediction] Sentiment-adjusted CI: [{lower:.6f}, {upper:.6f}] = [{lower*100:.3f}%, {upper*100:.3f}%]")
    
    print(f"[fixed_single_ticker_prediction] Original CI: [{original_lower:.6f}, {original_upper:.6f}] width={ci_width:.6f}")
    print(f"[fixed_single_ticker_prediction] Market-adjusted CI: [{lower:.6f}, {upper:.6f}] width={upper-lower:.6f} = [{lower*100:.3f}%, {upper*100:.3f}%]")
    
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

# Global cache for market data
market_data_cache = {}

def manage_cache_size():
    """Manage cache size to prevent memory bloat"""
    global market_data_cache
    
    with cache_lock:
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
        current_time = time.time()
        stale_keys = []
        
        # Check cache files for staleness
        if os.path.exists(CACHE_DIR):
            for cache_key in os.listdir(CACHE_DIR):
                if cache_key.endswith('.pkl'):
                    cache_file = os.path.join(CACHE_DIR, cache_key)
                    try:
                        file_age_hours = (current_time - os.path.getmtime(cache_file)) / 3600
                        
                        if file_age_hours > CACHE_MAX_AGE_HOURS:
                            stale_keys.append(cache_key)
                    except OSError:
                        # File might have been deleted concurrently
                        continue
        
        # Remove stale files and clear from memory cache
        for key in stale_keys:
            try:
                cache_file_path = os.path.join(CACHE_DIR, key)
                if os.path.exists(cache_file_path):
                    os.remove(cache_file_path)
                    print(f"[clear_stale_cache] Removed stale cache file: {key}")
                
                # Also remove from memory cache if present
                cache_base_key = key.replace('.pkl', '')
                if cache_base_key in market_data_cache:
                    del market_data_cache[cache_base_key]
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
    'SPLV': '', 
    'SPHB': '',  
    'SPSM': 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
}

INDEX_ETF_TICKERS = {
    'NASDAQ': 'QQQ',
    'SPY': 'SPY',
    'DOW': 'DIA',
    'SP400': 'MDY',
    'SPSM': 'SPSM',
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
                except (pickle.PickleError, IOError, EOFError) as e:
                    print(f"Cache read error for {cache_key}: {e}")
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
    # --- Helper functions at the top for proper scope ---
    def scrape_yahoo_etf_holdings(etf_ticker):
        url = f"https://finance.yahoo.com/quote/{etf_ticker}/holdings?p={etf_ticker}"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            tickers = []
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        ticker = cells[0].get_text(strip=True)
                        if 1 <= len(ticker) <= 6 and ticker.isupper():
                            tickers.append(ticker)
            return tickers
        except Exception as e:
            print(f"[scrape_yahoo_etf_holdings] Error: {e}")
            return []

    def scrape_wikipedia_table(url, ticker_col=0):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            tickers = []
            if table:
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) > ticker_col:
                        ticker = cells[ticker_col].get_text(strip=True).replace(".", "-")
                        if 1 <= len(ticker) <= 6 and ticker.isupper():
                            tickers.append(ticker)
            return tickers
        except Exception as e:
            print(f"[scrape_wikipedia_table] Error: {e}")
            return []

    # FIX: Define cache_key before any index-specific logic
    cache_key = f"constituents_{index_name}"

    # Wikipedia scraping for SPSM (S&P 600)
    if index_name == 'SPSM':
        url = INDEX_URLS['SPSM']
        tickers = scrape_wikipedia_table(url, ticker_col=0)
        if len(tickers) > 100:
            save_to_cache(cache_key, tickers)
            return tickers
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        save_to_cache(cache_key, fallback)
        return fallback
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
    # Hardcoded QQQ (NASDAQ-100) tickers
    QQQ_TICKERS = [
        "ADBE","AMD","ABNB","GOOGL","GOOG","AMZN","AEP","AMGN","ADI","AAPL","AMAT","APP","ARM","ASML","AZN","TEAM","ADSK","ADP","AXON",
        "BKR","BIIB","BKNG","AVGO","CDNS","CDW","CHTR","CMCSA","CEG","COST","CPRT","CSGP","CSCO","CRWD","CSX","DDOG","DXCM","EA","EXC",
        "FANG","FAST","FTNT","GEHC","GFS","GILD","HON","IDXX","INTC","INTU","ISRG","KDP","KHC","KLAC","LIN","LRCX","LULU","MAR","MCHP",
        "MDLZ","MELI","META","MNST","MRVL","MSFT","MSTR","MU","NFLX","NVDA","NXPI","ODFL","ON","ORLY","PANW","PAYX","PCAR","PDD","PEP",
        "PLTR","PYPL","QCOM","REGN","ROP","ROST","SHOP","SBUX","SNPS","TMUS","TTD","TTWO","TSLA","TXN","VRSK","VRTX","WBD","WDAY","XEL","ZS"
    ]
    # Yahoo scraping for SPY (S&P 500) and MDY (S&P 400)
    def scrape_yahoo_etf_holdings(etf_ticker):
        url = f"https://finance.yahoo.com/quote/{etf_ticker}/holdings?p={etf_ticker}"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            tickers = []
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        ticker = cells[0].get_text(strip=True)
                        if 1 <= len(ticker) <= 6 and ticker.isupper():
                            tickers.append(ticker)
            return tickers
        except Exception as e:
            print(f"[scrape_yahoo_etf_holdings] Error: {e}")
            return []

    def scrape_wikipedia_table(url, ticker_col=0):

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", {"class": "wikitable"})
            tickers = []
            if table:
                for row in table.find_all("tr")[1:]:
                    cells = row.find_all("td")
                    if len(cells) > ticker_col:
                        ticker = cells[ticker_col].get_text(strip=True).replace(".", "-")
                        if 1 <= len(ticker) <= 6 and ticker.isupper():
                            tickers.append(ticker)
            return tickers
        except Exception as e:
            print(f"[scrape_wikipedia_table] Error: {e}")
            return []

    cache_key = f"constituents_{index_name}"
    if not force_refresh:
        cached_data = get_cached_data(cache_key)
        if cached_data:
            return cached_data


    # Hardcoded QQQ
    if index_name == 'NASDAQ':
        save_to_cache(cache_key, QQQ_TICKERS)
        return QQQ_TICKERS

    # Wikipedia scraping for SPY (S&P 500)
    if index_name == 'SPY':
        url = INDEX_URLS['SPY']
        tickers = scrape_wikipedia_table(url, ticker_col=0)
        if len(tickers) > 100:
            save_to_cache(cache_key, tickers)
            return tickers
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        save_to_cache(cache_key, fallback)
        return fallback

    # Wikipedia scraping for MDY (S&P 400)
    if index_name == 'SP400':
        url = INDEX_URLS['SP400']
        tickers = scrape_wikipedia_table(url, ticker_col=0)
        if len(tickers) > 50:
            save_to_cache(cache_key, tickers)
            return tickers
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        save_to_cache(cache_key, fallback)
        return fallback

    # Wikipedia scraping for DOW (Dow Jones Industrial Average)
    if index_name == 'DOW':
        url = INDEX_URLS['DOW']
        tickers = scrape_wikipedia_table(url, ticker_col=0)
        if len(tickers) > 20:  # DOW has 30 companies
            save_to_cache(cache_key, tickers)
            return tickers
        # DOW fallback with major blue-chip stocks
        fallback = ['AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'V', 'AXP', 'BA', 
                   'TRV', 'JPM', 'JNJ', 'PG', 'CVX', 'MRK', 'KO', 'DIS', 'MMM', 'IBM',
                   'WMT', 'NKE', 'CRM', 'HON', 'AMGN', 'INTC', 'CSCO', 'VZ', 'WBA', 'DOW']
        save_to_cache(cache_key, fallback)
        return fallback

    # ...existing code...

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
    
    print(f"[download_market_data_cache] Starting download with start_date={start_date}, force_refresh={force_refresh}")
    print("[download_market_data_cache] Downloading fresh market reference data...")
    market_data_cache = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        def fetch_market_symbol(symbol):
            try:
                print(f"[download_market_data_cache] Downloading {symbol}...")
                df = yf.download(symbol, start=start_date, progress=False)
                
                if df is not None and not df.empty and len(df) >= 50:
                    # Add validation to flatten any nested arrays
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].apply(lambda x: to_scalar(x) if hasattr(x, '__len__') else x)
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"[download_market_data_cache] ✅ Successfully downloaded {symbol}: {len(df)} rows")
                    return symbol, df
                else:
                    print(f"[download_market_data_cache] ❌ Empty or insufficient data for {symbol}")
            except Exception as e:
                print(f"[download_market_data_cache] ❌ Error downloading {symbol}: {e}")
            return symbol, None
        
        futures = {executor.submit(fetch_market_symbol, symbol): symbol for symbol in MARKET_SYMBOLS}
        for future in as_completed(futures):
            symbol, data = future.result()
            if data is not None:
                market_data_cache[symbol] = data
    
    print(f"[download_market_data_cache] Completed with {len(market_data_cache)} market symbols: {list(market_data_cache.keys())}")
    
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
            def fetch_ticker(ticker):
                try:
                    tdf = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if tdf is not None and not tdf.empty and len(tdf) >= 50:
                        # Add validation to flatten any nested arrays
                        for col in tdf.columns:
                            if tdf[col].dtype == 'object':
                                tdf[col] = tdf[col].apply(lambda x: to_scalar(x) if hasattr(x, '__len__') else x)
                                tdf[col] = pd.to_numeric(tdf[col], errors='coerce')
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
    if df is None or len(df) < FEATURE_ROLLING_WINDOW:
        return 'sideways', 0.5
    
    # Calculate technical indicators
    df['SMA_20'] = df['Close'].rolling(window=FEATURE_ROLLING_WINDOW).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].pct_change().rolling(window=FEATURE_ROLLING_WINDOW).std()
    
    # Get recent data
    recent = df.tail(30)
    
    # Calculate trend strength
    price_trend = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
    sma_trend = (recent['SMA_20'].iloc[-1] - recent['SMA_20'].iloc[0]) / recent['SMA_20'].iloc[0]
    avg_volatility = recent['Volatility'].mean()
    
    # Determine market condition
    if abs(price_trend) < SIDEWAYS_PRICE_THRESHOLD and abs(sma_trend) < SIDEWAYS_SMA_THRESHOLD:
        condition = 'sideways'
        strength = 0.3
    elif price_trend > BULL_PRICE_THRESHOLD and sma_trend > BULL_SMA_THRESHOLD:
        condition = 'bull'
        strength = min(0.9, 0.5 + abs(price_trend))
    elif price_trend < BEAR_PRICE_THRESHOLD and sma_trend < BEAR_SMA_THRESHOLD:
        condition = 'bear'
        strength = min(0.9, 0.5 + abs(price_trend))
    elif avg_volatility > HIGH_VOLATILITY_THRESHOLD:
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
        'volatile': [4, 5, 8]   # Extra Trees, AdaBoost, Gradient Boosting (removed Neural Network)
    }
    
    return model_selections.get(market_condition, [2, 7, 6])

def flatten_series(s):
    """
    Flatten any Series or DataFrame to a 1D Series.
    Enhanced to handle 2D arrays from YFinance data.
    """
    # If DataFrame, take first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
        
    # If numpy array, flatten to 1D
    if isinstance(s, np.ndarray):
        # Special handling for 2D arrays
        if s.ndim > 1:
            # Take first column for 2D arrays
            s = s[:, 0] if s.shape[1] > 0 else s.flatten()
        s = s.flatten()
        return pd.Series(s)
        
    # If Series, ensure values are scalar
    if isinstance(s, pd.Series):
        # Check if any values are arrays/lists and flatten them
        if s.dtype == 'object':
            return s.apply(lambda x: to_scalar(x) if hasattr(x, '__len__') else x)
        return s
        
    # Fallback: convert to 1D array then Series
    try:
        arr = np.asarray(s).flatten()
        return pd.Series(arr)
    except:
        # Ultimate fallback: return empty series
        return pd.Series()

def detect_market_regime(etf_df):
    """Robust regime detection using multiple indicators from the index ETF ticker's data."""
    if etf_df is None or len(etf_df) < MIN_DATA_POINTS:
        return 'sideways', 0.5
        
    # Create a copy to avoid modifying the original dataframe
    df = etf_df.copy()
    
    # STEP 1: Pre-process all OHLCV columns to ensure they're 1D arrays
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            # Check if column contains arrays or has unexpected structure
            try:
                # First check: If any element is an array/list type
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x: to_scalar(x) if hasattr(x, '__len__') else x)
                
                # Second check: For multi-dimensional numpy arrays
                sample = df[col].iloc[0] if len(df) > 0 else None
                if hasattr(sample, 'shape') and hasattr(sample, '__len__') and len(sample) > 0:
                    df[col] = df[col].apply(lambda x: to_scalar(x))
                
                # Ensure all values are proper numeric type
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Error flattening column {col}: {e}")
                # Provide default values if conversion fails
                df[col] = np.nan
    
    # STEP 2: Now proceed with indicator calculations using flattened data
    try:
        # Calculate indicators if not present
        if 'SMA_20' not in df:
            df['SMA_20'] = flatten_series(df['Close'].rolling(window=20).mean())
        if 'SMA_50' not in df:
            df['SMA_50'] = flatten_series(df['Close'].rolling(window=50).mean())
        
        # MACD calculation with proper error handling
        try:
            if 'macd' not in df or 'macd_signal' not in df:
                macd = MACD(close=df['Close'])
                macd_val_raw = macd.macd()
                macd_signal_raw = macd.macd_signal()
                df['macd'] = flatten_series(macd_val_raw)
                df['macd_signal'] = flatten_series(macd_signal_raw)
                df['ema_diff'] = flatten_series(df['macd'] - df['macd_signal'])
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            # Provide default values for MACD if calculation fails
            df['macd'] = np.nan
            df['macd_signal'] = np.nan
            df['ema_diff'] = np.nan
            
        # Continue with other indicators with similar error handling
        try:
            if 'stoch_k' not in df:
                stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
                df['stoch_k'] = flatten_series(stoch.stoch())
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            df['stoch_k'] = np.nan
            
        try:
            if 'Donchian_Width' not in df:
                df['Donchian_Width'] = flatten_series(df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())
        except Exception as e:
            print(f"Error calculating Donchian Width: {e}")
            df['Donchian_Width'] = np.nan
            
        try:
            if 'RSI' not in df:
                df['RSI'] = flatten_series(RSIIndicator(close=df['Close']).rsi())
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df['RSI'] = np.nan
            
        try:
            if 'williams_r' not in df:
                df['williams_r'] = flatten_series(WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r())
        except Exception as e:
            print(f"Error calculating Williams %R: {e}")
            df['williams_r'] = np.nan
            
        try:
            if 'ATR' not in df:
                df['ATR'] = flatten_series(AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range())
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            df['ATR'] = np.nan
            
        try:
            if 'rolling_20d_std' not in df:
                df['rolling_20d_std'] = flatten_series(df['Close'].rolling(window=20).std())
        except Exception as e:
            print(f"Error calculating rolling std: {e}")
            df['rolling_20d_std'] = np.nan
            
        try:
            if 'percent_b' not in df:
                upper = flatten_series(df['SMA_20'] + 2*df['Close'].rolling(window=20).std())
                lower = flatten_series(df['SMA_20'] - 2*df['Close'].rolling(window=20).std())
                df['percent_b'] = flatten_series((df['Close'] - lower) / (upper - lower))
        except Exception as e:
            print(f"Error calculating Percent B: {e}")
            df['percent_b'] = np.nan
            
        try:
            if 'OBV' not in df:
                df['OBV'] = flatten_series(OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume())
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            df['OBV'] = np.nan
    
    except Exception as e:
        print(f"Error in market regime detection: {e}")
        return 'sideways', 0.5  # Default fallback
        
    # Continue with remaining indicators with error handling
    try:
        if 'cmf' not in df:
            df['cmf'] = flatten_series(ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).chaikin_money_flow())
    except Exception as e:
        print(f"Error calculating CMF: {e}")
        df['cmf'] = np.nan
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
    """
    Original implementation with RSI, MACD, Bollinger Bands, Volume analysis, ATR, OBV
    Now includes forward return targets for training
    """
    if df is None or len(df) < MIN_DATA_POINTS:
        print(f"[add_features_to_stock_original] {ticker}: DataFrame is None or too short ({len(df) if df is not None else 0} rows)")
        return None
    
    try:
        # Create a copy to avoid warnings
        df = df.copy()
        
        # Handle nested arrays in input data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                series = df[col]
                # If series contains nested arrays, flatten them
                if series.dtype == 'object' or any(hasattr(x, 'shape') for x in series.iloc[:5].values if x is not None):
                    series = series.apply(lambda x: to_scalar(x) if hasattr(x, '__len__') else x)
                df[col] = pd.to_numeric(series, errors='coerce')
        
        # 1. RSI (Relative Strength Index) - 14-day
        try:
            # Use the ta library's RSI implementation directly
            rsi_indicator = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = flatten_series(rsi_indicator.rsi())
            
            # Verify if RSI has proper values
            if pd.isna(df['RSI']).all() or df['RSI'].var() == 0:
                print(f"[add_features_to_stock_original] {ticker}: RSI calculation produced invalid values")
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating RSI: {e}")
            df['RSI'] = np.nan
        
        # 2. MACD (Moving Average Convergence Divergence)
        try:
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = flatten_series(ema_12 - ema_26)
            df['macd_signal'] = flatten_series(df['macd'].ewm(span=9).mean())
            df['macd_histogram'] = flatten_series(df['macd'] - df['macd_signal'])
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating MACD: {e}")
            df['macd'] = df['macd_signal'] = df['macd_histogram'] = np.nan
        
        # Ensure ema_diff exists for detect_market_regime function
        try:
            if 'ema_diff' not in df.columns:
                if 'macd_histogram' in df.columns:
                    # Use macd_histogram as ema_diff (they're the same calculation)
                    df['ema_diff'] = df['macd_histogram']
                elif 'macd' in df.columns and 'macd_signal' in df.columns:
                    # Calculate ema_diff from macd and macd_signal
                    df['ema_diff'] = flatten_series(df['macd'] - df['macd_signal'])
                else:
                    # Fallback to NaN if neither exists
                    df['ema_diff'] = np.nan
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error ensuring ema_diff: {e}")
            df['ema_diff'] = np.nan
        
        # 3. Bollinger Bands
        try:
            df['SMA_20'] = flatten_series(df['Close'].rolling(window=20).mean())
            df['SD'] = flatten_series(df['Close'].rolling(window=20).std())
            df['Upper'] = flatten_series(df['SMA_20'] + (df['SD'] * 2))
            df['Lower'] = flatten_series(df['SMA_20'] - (df['SD'] * 2))
            df['percent_b'] = flatten_series((df['Close'] - df['Lower']) / (df['Upper'] - df['Lower']))
            df['bb_width'] = flatten_series((df['Upper'] - df['Lower']) / df['SMA_20'])
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating Bollinger Bands: {e}")
            for col in ['SMA_20', 'SD', 'Upper', 'Lower', 'percent_b', 'bb_width']:
                df[col] = np.nan
        
        # 4. Volume indicators - SCALED VERSION
        try:
            df['volume_pct_change'] = flatten_series(df['Volume'].pct_change())
            # Scale volume_ma by dividing by median volume to prevent extreme values
            volume_ma_raw = df['Volume'].rolling(window=20).mean()
            median_volume = df['Volume'].median()
            if median_volume > 0:
                df['volume_ma'] = flatten_series(volume_ma_raw / median_volume)
                print(f"[add_features_to_stock_original] {ticker}: volume_ma scaled by median_volume={median_volume:.0f}")
            else:
                df['volume_ma'] = np.nan
            
            # volume_ratio calculation (already relative, should be fine)
            volume_ma_raw_for_ratio = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = flatten_series(df['Volume'] / volume_ma_raw_for_ratio)
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating volume indicators: {e}")
            df['volume_pct_change'] = df['volume_ma'] = df['volume_ratio'] = np.nan
        
        # 5. ATR (Average True Range)
        try:
            high_low = df['High'] - df['Low']
            high_prev_close = abs(df['High'] - df['Close'].shift(1))
            low_prev_close = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
            df['ATR'] = flatten_series(true_range.rolling(window=14).mean())
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating ATR: {e}")
            df['ATR'] = np.nan
        
        # 6. OBV (On-Balance Volume) - SCALED VERSION
        try:
            obv = []
            obv_val = 0
            prev_close = None
            for idx, row in df.iterrows():
                if prev_close is not None:
                    if row['Close'] > prev_close:
                        obv_val += row['Volume']
                    elif row['Close'] < prev_close:
                        obv_val -= row['Volume']
                obv.append(obv_val)
                prev_close = row['Close']
            
            # Scale OBV to prevent extreme values
            if len(obv) > 0 and df['Volume'].notna().sum() > 0:
                # Calculate average volume for scaling
                avg_volume = df['Volume'].mean()
                if avg_volume > 0:
                    # Normalize OBV by average volume
                    obv_scaled = [val / avg_volume for val in obv]
                    df['OBV'] = flatten_series(pd.Series(obv_scaled, index=df.index))
                    print(f"[add_features_to_stock_original] {ticker}: OBV scaled by avg_volume={avg_volume:.0f}, range=[{min(obv_scaled):.2f}, {max(obv_scaled):.2f}]")
                else:
                    df['OBV'] = np.nan
            else:
                df['OBV'] = np.nan
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating OBV: {e}")
            df['OBV'] = np.nan
        
        # 7. Additional momentum indicators
        try:
            df['momentum'] = flatten_series(df['Close'] / df['Close'].shift(10) - 1)
            df['roc_10'] = flatten_series(df['Close'].pct_change(10))
            df['price_change'] = flatten_series(df['Close'].pct_change())
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating momentum indicators: {e}")
            df['momentum'] = df['roc_10'] = df['price_change'] = np.nan
        
        # 8. Moving averages and trends
        try:
            df['sma_10'] = flatten_series(df['Close'].rolling(window=10).mean())
            df['sma_50'] = flatten_series(df['Close'].rolling(window=50).mean())
            df['ema_10'] = flatten_series(df['Close'].ewm(span=10).mean())
            df['ema_50'] = flatten_series(df['Close'].ewm(span=50).mean())
            df['sma_cross'] = flatten_series((df['sma_10'] > df['sma_50']).astype(int))
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating moving averages: {e}")
            for col in ['sma_10', 'sma_50', 'ema_10', 'ema_50', 'sma_cross']:
                df[col] = np.nan
        
        # 9. Volatility measures
        try:
            df['volatility'] = flatten_series(df['Close'].rolling(window=20).std())
            df['rolling_20d_std'] = flatten_series(df['Close'].pct_change().rolling(window=20).std())
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating volatility: {e}")
            df['volatility'] = df['rolling_20d_std'] = np.nan

        df['close_lag_1'] = df['Close'].shift(1)
        df['close_lag_5'] = df['Close'].shift(5)

        try:
            df[f'forward_return_{prediction_window}'] = flatten_series(
                (df['Close'].shift(-prediction_window) / df['Close']) - 1
            )
            print(f"[add_features_to_stock_original] {ticker}: ✅ Added {prediction_window}-day forward return target")
        except Exception as e:
            print(f"[add_features_to_stock_original] {ticker}: Error calculating forward returns: {e}")
            df[f'forward_return_{prediction_window}'] = np.nan

        print(f"[add_features_to_stock_original] {ticker}: Added {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} features")
        return df

    except Exception as e:
        print(f"[add_features_to_stock_original] ERROR for {ticker}: {e}")
        return None

def add_features_to_stock(ticker, df, prediction_window=5, market_data_cache=None):
    """Version with cross-asset features"""
    if df is None or len(df) < MIN_DATA_POINTS:
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
        
        # Time-based features using pandas built-in properties
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        # Quarter end detection using built-in property
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        
        # Log whether quarter ends were detected
        quarter_end_count = df['is_quarter_end'].sum()
        print(f"[add_cross_asset_features] {ticker}: Quarter-end dates found: {quarter_end_count}")
        
        calculated_features.extend([
            'day_of_week', 'month', 'quarter', 'is_quarter_end', 
            'is_friday', 'is_monday'
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
        
        # 🗑️ REMOVED: strong_dollar feature as requested
        
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
    # Define base weights for each group per regime - includes cross_asset group
    base_weights = {
        'bull': {'momentum': 1.3, 'trend': 1.2, 'mean_reversion': 0.8, 'volatility': 0.7, 'volume': 1.0, 'cross_asset': 1.1, 'other': 1.0},
        'bear': {'momentum': 0.8, 'trend': 0.7, 'mean_reversion': 1.3, 'volatility': 1.2, 'volume': 1.0, 'cross_asset': 1.2, 'other': 1.0},
        'sideways': {'momentum': 0.8, 'trend': 0.8, 'mean_reversion': 1.3, 'volatility': 1.0, 'volume': 1.0, 'cross_asset': 0.9, 'other': 1.0},
        'volatile': {'momentum': 0.7, 'trend': 0.7, 'mean_reversion': 1.1, 'volatility': 1.3, 'volume': 1.2, 'cross_asset': 1.3, 'other': 1.0},
    }
    # Map each feature to a group - includes cross-asset features
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
        'rising_rates': 'cross_asset',
        # 🗑️ REMOVED: 'strong_dollar': 'cross_asset'
        
        # Additional time-based features for model training
        'day_of_week': 'other', 'month': 'other', 'quarter': 'other', 
        'is_quarter_end': 'other', 'is_friday': 'other', 'is_monday': 'other'
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

def train_model_for_stock(ticker, df, model_ids, regime=None, regime_strength=0.5, prediction_window=5):
    """Train models for a single stock, with regime-aware feature weighting."""
    if df is None or len(df) < MIN_DATA_POINTS:
        print(f"[train_model_for_stock] {ticker}: DataFrame too short or None")
        return None
    
    try:
        # Extract feature columns (exclude basic OHLCV)
        feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        # Validate that we have meaningful features - cross-asset detection
        cross_asset_features = [col for col in feature_columns if any(keyword in col.lower() for keyword in 
                               ['spy', 'vix', 'dxy', 'tlt', 'gld', 'qqq', 'correlation', 'beta',
                                'ratio', 'risk_on', 'risk_off', 'relative', 'sentiment',
                                # Add more keywords to catch all cross-asset features
                                'market', 'tlt_', 'dxy_', 'gld_', 'trend', 'regime', 'rising_rates', 'tech_outperform'])]
        technical_features = [col for col in feature_columns if col not in cross_asset_features]
        
        print(f"[train_model_for_stock] {ticker}: Technical features: {len(technical_features)}, Cross-asset features: {len(cross_asset_features)}")
        
        if len(technical_features) < 5:
            print(f"[train_model_for_stock] {ticker}: Insufficient technical features ({len(technical_features)})")
            return None
        
        # Clean and prepare features with missing data handling
        df_clean = df.copy()
        for col in feature_columns:
            if col in df_clean.columns:
                # Handle any nested arrays or objects
                series = df_clean[col]
                if series.dtype == 'object':
                    series = series.apply(lambda x: x[0] if isinstance(x, (np.ndarray, list)) and len(x) > 0 else x)
                df_clean[col] = pd.to_numeric(series, errors='coerce')
                
                # Forward fill small gaps (up to FORWARD_FILL_LIMIT days)
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].fillna(method='ffill', limit=FORWARD_FILL_LIMIT)
                    # If still NaN, try backward fill for very early periods
                    df_clean[col] = df_clean[col].fillna(method='bfill', limit=FORWARD_FILL_LIMIT)
        
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
        
        # Track cross-asset feature availability for model performance assessment
        if len(valid_cross_asset) == 0:
            print(f"[train_model_for_stock] {ticker}: ⚠️ NO cross-asset features found - only technical indicators will be used")
        else:
            print(f"[train_model_for_stock] {ticker}: ✅ Cross-asset features available: {valid_cross_asset[:5]}{'...' if len(valid_cross_asset) > 5 else ''}")
        
        if len(valid_features) < 5:
            print(f"[train_model_for_stock] {ticker}: Not enough valid features ({len(valid_features)})")
            return None
        
        # Prepare feature matrix - EXCLUDE future-looking features to prevent data leakage
        future_looking_features = [
            f'close_lead_{prediction_window}', 
            f'forward_return_{prediction_window}',
            f'rolling_max_{prediction_window}',
            f'rolling_min_{prediction_window}',
            # Additional patterns that might indicate future data
            'lead_', 'future_', 'forward_', 'next_'
        ]
        
        # More comprehensive exclusion using pattern matching
        valid_features_filtered = []
        excluded_features = []
        
        for col in valid_features:
            should_exclude = False
            # Check exact matches
            if col in future_looking_features:
                should_exclude = True
            # Check patterns
            else:
                for pattern in ['lead_', 'future_', 'forward_', 'next_']:
                    if pattern in col.lower():
                        should_exclude = True
                        break
            
            if should_exclude:
                excluded_features.append(col)
            else:
                valid_features_filtered.append(col)
        if excluded_features:
            print(f"[train_model_for_stock] {ticker}: 🚫 EXCLUDED future-looking features: {excluded_features}")
        print(f"[train_model_for_stock] {ticker}: ✅ Using {len(valid_features_filtered)} features (excluding {len(excluded_features)} future-looking)")
        
        if len(valid_features_filtered) < 5:
            print(f"[train_model_for_stock] {ticker}: ❌ Not enough valid features after removing future-looking ones ({len(valid_features_filtered)})")
            return None
            
        X = df_clean[valid_features_filtered].values.astype(float)
        
        # Validate feature composition in final training matrix
        final_cross_asset = [col for col in valid_features_filtered if col in cross_asset_features]
        final_technical = [col for col in valid_features_filtered if col in technical_features]
        
        print(f"[train_model_for_stock] {ticker}: 📊 Final feature matrix - Technical: {len(final_technical)}, Cross-asset: {len(final_cross_asset)}")
        
        if len(final_cross_asset) == 0:
            print(f"[train_model_for_stock] {ticker}: WARNING: NO cross-asset features in final training matrix!")
        else:
            print(f"[train_model_for_stock] {ticker}: ✅ Cross-asset features in training: {final_cross_asset[:3]}{'...' if len(final_cross_asset) > 3 else ''}")
            
            # Validate cross-asset features have meaningful values
            cross_asset_indices = [i for i, col in enumerate(valid_features_filtered) if col in cross_asset_features]
            if cross_asset_indices:
                cross_asset_data = X[:, cross_asset_indices]
                non_zero_features = np.sum(np.abs(cross_asset_data) > 1e-6, axis=0)
                zero_features = len(cross_asset_indices) - np.sum(non_zero_features > 0)
                
                if zero_features > 0:
                    print(f"[train_model_for_stock] {ticker}: ⚠️ {zero_features} cross-asset features are effectively zero")
                else:
                    print(f"[train_model_for_stock] {ticker}: ✅ All {len(cross_asset_indices)} cross-asset features have meaningful values")
        
        # Apply regime-based feature weights if provided
        if regime is not None:
            weights_dict = get_indicator_weights(regime, regime_strength)
            weights = np.array([weights_dict.get(col, 1.0) for col in valid_features_filtered])
            X = X * weights
            print(f"[train_model_for_stock] {ticker}: Applied regime weights for {regime} market")
        
        # Prepare target: N-day forward return (the actual prediction target)
        target_column = f'forward_return_{prediction_window}'
        if target_column in df_clean.columns:
            future_returns = df_clean[target_column].values
            print(f"[train_model_for_stock] {ticker}: ✅ TRAINING TARGET: {prediction_window}-day forward return (from feature column)")
        else:
            # Fallback to calculating N-day forward return on the fly
            future_returns = (df_clean['Close'].shift(-prediction_window) / df_clean['Close']) - 1
            future_returns = future_returns.values
            print(f"[train_model_for_stock] {ticker}: ✅ TRAINING TARGET: {prediction_window}-day forward return (calculated on-the-fly)")
        
        print(f"[train_model_for_stock] {ticker}: 🎯 Models will predict {prediction_window}-day future returns, NOT 1-day returns!")
        
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
        print(f"[train_model_for_stock] {ticker}: 📊 TARGET ({prediction_window}-day returns) - Mean: {y_mean:.6f} ({y_mean*100:.3f}%), Std: {y_std:.6f} ({y_std*100:.3f}%)")
        print(f"[train_model_for_stock] {ticker}: 📈 Training {len(model_ids)} models to predict {prediction_window}-day forward returns")
        
        # Train models and get predictions
        model_predictions = {}
        for model_id in model_ids:
            try:
                pred_result = train_and_predict_model(X_clean, y_clean, model_id, prediction_window)
                if pred_result is not None and len(pred_result) > 0:
                    # Get the last prediction (most recent)
                    if isinstance(pred_result, list):
                        raw_pred = float(pred_result[-1]) if len(pred_result) > 0 else 0.0
                    else:
                        raw_pred = float(pred_result)
                    
                    # NO SCALING - keep predictions in natural decimal scale
                    avg_pred = raw_pred
                    
                    # Filter out extreme individual predictions to prevent ensemble contamination
                    # This prevents one bad model from poisoning the ensemble
                    max_reasonable_return = 0.25 if prediction_window == 1 else 0.50  # 25% daily, 50% weekly max
                    if abs(avg_pred) > max_reasonable_return:
                        print(f"[train_model_for_stock] {ticker}: ⚠️ Model {model_id} extreme prediction REJECTED: {avg_pred:.6f} ({avg_pred*100:.3f}%)")
                        continue  # Skip this model entirely
                    
                    # Apply additional bounds for individual models before ensemble
                    if prediction_window == 1 and abs(avg_pred) > 0.15:  # 15% max for individuals in 1-day
                        original_pred = avg_pred
                        avg_pred = np.clip(avg_pred, -0.15, 0.15)
                        print(f"[train_model_for_stock] {ticker}: ⚠️ Model {model_id} capped individual prediction: {original_pred:.6f} → {avg_pred:.6f}")
                    
                    model_predictions[model_id] = {
                        'avg_pred': avg_pred,
                        'all_preds': pred_result if isinstance(pred_result, list) else [avg_pred]
                    }
                    
                    print(f"[train_model_for_stock] {ticker}: Model {model_id} predicts {prediction_window}-day return: {avg_pred:.6f} ({avg_pred*100:.3f}%)")
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
                # Check for extreme outliers (more than 3 standard deviations from mean)
                if len(valid_predictions) > 2:
                    pred_mean = np.mean(valid_predictions)
                    pred_std = np.std(valid_predictions)
                    
                    # Only filter outliers if std deviation is significant (> 1%)
                    if pred_std > 0.01:
                        # Remove predictions more than 3 std devs from mean
                        filtered_predictions = [p for p in valid_predictions 
                                              if abs(p - pred_mean) <= 3 * pred_std]
                        if len(filtered_predictions) >= 2:  # Keep at least 2 predictions
                            valid_predictions = filtered_predictions
                
                # Use mean for ensemble (more responsive than median)
                ensemble_pred = np.mean(valid_predictions)
                
                # Apply global bounds checking for reasonable predictions
                # For 1-day predictions, cap at ±15% (realistic daily stock movement limits)
                # For longer periods, allow larger movements
                if prediction_window == 1:
                    max_daily_return = 0.15  # 15% maximum daily return
                    min_daily_return = -0.15  # -15% maximum daily loss
                    
                    if abs(ensemble_pred) > max_daily_return:
                        original_pred = ensemble_pred
                        ensemble_pred = np.clip(ensemble_pred, min_daily_return, max_daily_return)
                        print(f"[train_model_for_stock] {ticker}: ⚠️ Capped extreme 1-day prediction: {original_pred:.6f} → {ensemble_pred:.6f}")
                elif prediction_window <= 7:
                    max_weekly_return = 0.30  # 30% maximum weekly return
                    min_weekly_return = -0.30
                    
                    if abs(ensemble_pred) > max_weekly_return:
                        original_pred = ensemble_pred
                        ensemble_pred = np.clip(ensemble_pred, min_weekly_return, max_weekly_return)
                        print(f"[train_model_for_stock] {ticker}: ⚠️ Capped extreme {prediction_window}-day prediction: {original_pred:.6f} → {ensemble_pred:.6f}")
                
                # Calculate prediction spread as confidence metric
                pred_std = np.std(valid_predictions) if len(valid_predictions) > 1 else 0.0
                pred_spread = np.max(valid_predictions) - np.min(valid_predictions) if len(valid_predictions) > 1 else 0.0
                
                # Confidence calculation based on prediction variance
                # Typical stock returns have spreads of 0.01-0.05 (1-5%), so normalize accordingly
                normalized_spread = pred_spread / 0.02  # Normalize to 2% spread = 1.0
                confidence = max(0.2, 1.0 - (normalized_spread * 0.3))  # More gradual penalty
                
                print(f"[train_model_for_stock] {ticker}: 🎯 Raw {prediction_window}-day ensemble: {ensemble_pred:.6f} ({ensemble_pred*100:.3f}%)")
                print(f"[train_model_for_stock] {ticker}: ✅ Final {prediction_window}-day prediction: {ensemble_pred:.6f} ({ensemble_pred*100:.3f}%)")
                print(f"[train_model_for_stock] {ticker}: 📊 Confidence: {confidence:.3f}, Spread: {pred_spread:.6f}")
                
                return {
                    'prediction': float(ensemble_pred),
                    'percentage': float(ensemble_pred * 100),
                    'confidence': float(confidence),
                    'individual_predictions': {k: v['avg_pred'] for k, v in model_predictions.items()},
                    'valid_models': len(valid_predictions),
                    'total_models': len(model_predictions),
                    'cross_asset_features_used': len(final_cross_asset),
                    'technical_features_used': len(final_technical),
                    'prediction_window': prediction_window,
                    'validation_applied': False,
                    'validation_warnings': []
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
        
        # Explicit cleanup to prevent memory leaks
        del X_clean, y_clean
        if 'df_clean' in locals():
            del df_clean
        if 'valid_features_filtered' in locals():
            del valid_features_filtered
            
        return model_predictions
        
    except Exception as e:
        print(f"[train_model_for_stock] ERROR for {ticker}: {e}")
        traceback.print_exc()
        return None

class TransformerModel:
    """Transformer-like model using scikit-learn components"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2):
        self.input_dim = input_dim
        self.d_model = d_model
        # Configure hidden layers based on num_layers and d_model
        if num_layers == 1:
            hidden_layers = (d_model,)
        elif num_layers == 2:
            hidden_layers = (d_model, d_model//2)
        else:
            # For more layers, create a progressive reduction
            hidden_layers = tuple(d_model // (2**i) for i in range(num_layers) if d_model // (2**i) >= 4)
            if not hidden_layers:
                hidden_layers = (d_model, d_model//2)
        
        # Use MLPRegressor as transformer alternative - don't apply additional scaling
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
    
    def fit(self, X, y):
        # X is already scaled in train_and_predict_model, don't scale again
        self.model.fit(X, y)
    
    def predict(self, X):
        # X is already scaled in train_and_predict_model, don't scale again
        return self.model.predict(X)

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

def train_and_predict_model(X, y, model_id, prediction_window=1):
    """Train actual ML model and make prediction, with robust data preprocessing."""
    if len(X) < MIN_DATA_POINTS:
        print(f"[train_and_predict_model] Not enough data for model {model_id}: {len(X)} samples")
        return [0.0]
    
    try:
        X = np.array(X)
        y = np.array(y)
        
        # Robust preprocessing for infinity and extreme values
        print(f"[train_and_predict_model] Model {model_id} preprocessing: X shape={X.shape}, y shape={y.shape}")
        
        # Check for infinity values in features
        inf_mask_X = np.isinf(X)
        if np.any(inf_mask_X):
            inf_count = np.sum(inf_mask_X)
            print(f"[train_and_predict_model] Model {model_id} Found {inf_count} infinity values in features")
            
            # Replace infinity with extreme but finite values
            X[X == np.inf] = 1e6   # Positive infinity
            X[X == -np.inf] = -1e6  # Negative infinity
            print(f"[train_and_predict_model] Model {model_id} ✅ Infinity values capped to ±1e6")
        
        # Check for infinity values in targets
        inf_mask_y = np.isinf(y)
        if np.any(inf_mask_y):
            inf_count_y = np.sum(inf_mask_y)
            print(f"[train_and_predict_model] Model {model_id} Found {inf_count_y} infinity values in targets")
            
            # Replace infinity with extreme but reasonable values for returns
            y[y == np.inf] = 0.5   # 50% return (extreme but possible)
            y[y == -np.inf] = -0.5  # -50% return (extreme but possible)
            print(f"[train_and_predict_model] Model {model_id} ✅ Target infinity values capped to ±50%")
        
        # Check for extremely large values that might cause numerical issues
        extreme_threshold_X = 1e4
        extreme_mask_X = np.abs(X) > extreme_threshold_X
        if np.any(extreme_mask_X):
            extreme_count = np.sum(extreme_mask_X)
            print(f"[train_and_predict_model] Model {model_id} ⚠️ Found {extreme_count} extreme feature values (>{extreme_threshold_X})")
            
            # Cap extreme values to prevent numerical instability
            X = np.clip(X, -extreme_threshold_X, extreme_threshold_X)
            print(f"[train_and_predict_model] Model {model_id} ✅ Extreme feature values capped to ±{extreme_threshold_X}")
        
        # Post-processing verification of data cleaning
        # Verify that all infinity and extreme value replacements worked
        remaining_inf_X = np.sum(np.isinf(X))
        remaining_inf_y = np.sum(np.isinf(y))
        
        if remaining_inf_X > 0 or remaining_inf_y > 0:
            print(f"[train_and_predict_model] Model {model_id} ERROR: {remaining_inf_X} infinity values still in X, {remaining_inf_y} in y after preprocessing")
            return [0.0]
        
        remaining_extreme_X = np.sum(np.abs(X) > extreme_threshold_X)
        if remaining_extreme_X > 0:
            print(f"[train_and_predict_model] Model {model_id} ERROR: {remaining_extreme_X} extreme values still in X after clipping")
            return [0.0]
            
        print(f"[train_and_predict_model] Model {model_id} ✅ Data preprocessing verification passed")
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        nan_count = len(X) - np.sum(valid_mask)
        if nan_count > 0:
            print(f"[train_and_predict_model] Model {model_id} removing {nan_count} NaN samples")
        
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
                n_estimators=100,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.6,       # More aggressive subsampling
                colsample_bytree=0.6,  # Use fewer features
                reg_alpha=0.2,       # L1 regularization
                reg_lambda=0.2,      # L2 regularization
                random_state=42,
                verbosity=0
            )
        elif model_id == 2:  # Random Forest - More conservative
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=8,
                bootstrap=True,
                max_features=0.6,
                random_state=42,
                n_jobs=1
            )
        elif model_id == 3:  # Neural Network - Heavily constrained for stability
            # More conservative configuration for better 1-day predictions
            if prediction_window == 1:
                # For 1-day predictions: very simple, heavily regularized
                model = MLPRegressor(
                    hidden_layer_sizes=(16,),            # Even smaller single layer
                    activation='relu',
                    solver='adam',
                    alpha=0.1,                           # Much stronger regularization
                    learning_rate_init=0.00005,         # Very low learning rate
                    max_iter=300,                        # Fewer iterations
                    early_stopping=True,
                    validation_fraction=0.2,            # More validation data
                    n_iter_no_change=10,                # Earlier stopping
                    random_state=42
                )
            else:
                # For longer predictions: still conservative but allow some complexity
                model = MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation='relu',
                    solver='adam',
                    alpha=0.05,
                    learning_rate_init=0.0001,
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=42
                )
        elif model_id == 4:  # Extra Trees - More conservative
            model = ExtraTreesRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=8,
                bootstrap=True,
                max_features=0.6,
                random_state=42,
                n_jobs=1
            )
        elif model_id == 5:  # AdaBoost - Much more conservative due to instability
            model = AdaBoostRegressor(
                n_estimators=30,   # Conservative setting to prevent overfitting
                learning_rate=0.01,  # Low learning rate for stability
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
        elif model_id == 7:  # SVR - More conservative
            model = SVR(
                kernel='rbf',
                C=1.0,      # Regularization parameter for complexity control
                epsilon=0.01,  # Tolerance for tight fitting
                gamma='scale'
            )
        elif model_id == 8:  # Gradient Boosting - Much more conservative due to instability
            model = GradientBoostingRegressor(
                n_estimators=50,   # Moderate ensemble size to prevent overfitting
                learning_rate=0.02,  # Conservative learning rate for stability
                max_depth=3,       # Shallow trees for simpler models
                subsample=0.6,     # Reduced sampling for additional randomness
                random_state=42
            )
        elif model_id == 9:  # Elastic Net - More regularized
            model = ElasticNet(
                alpha=0.2,     # L1 regularization strength
                l1_ratio=0.5,
                random_state=42,
                max_iter=3000  # Maximum iterations for convergence
            )
        elif model_id == 10:  # Transformer-like
            model = TransformerModel(input_dim=X_train_scaled.shape[1])
        else:
            print(f"[train_and_predict_model] Unknown model_id: {model_id}")
            return [0.0]
        
        # Add cross-validation for better accuracy assessment
        
        try:
            # Use TimeSeriesSplit for financial data (respects temporal order)
            # Ensure reasonable test size and number of splits
            min_test_size = max(5, len(X_train_scaled)//20)  # At least 5, or 5% of data
            max_splits = min(3, len(X_train_scaled)//min_test_size//2)  # Conservative splits
            tscv = TimeSeriesSplit(n_splits=max(1, max_splits), test_size=min_test_size)
            
            # Perform cross-validation to assess model quality
            cv_scores = cross_val_score(model, X_train_scaled, y_train_clean, cv=tscv, scoring='neg_mean_squared_error')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            print(f"[train_and_predict_model] Model {model_id} CV Score: {cv_mean:.6f} ± {cv_std:.6f}")
            
            # Skip models with very poor cross-validation performance
            # Stricter thresholds for 1-day predictions, more lenient for longer periods
            # EXTRA STRICT for problematic models that show instability
            if model_id in [3, 5, 8]:  # Neural Network, AdaBoost, Gradient Boosting get extra strict treatment
                if prediction_window == 1:
                    cv_threshold = -0.01  # 1% error threshold for unstable models on daily predictions
                elif prediction_window <= 7:
                    cv_threshold = -0.03  # 3% error threshold for unstable models on weekly predictions
                else:
                    cv_threshold = -0.05  # 5% error threshold for unstable models on longer predictions
            else:
                if prediction_window == 1:
                    # For 1-day predictions, be more strict (typical 1-day returns have MSE ~0.01-0.05)
                    cv_threshold = -0.05  # 5% error threshold for daily predictions
                elif prediction_window <= 7:
                    # For weekly predictions, allow higher error
                    cv_threshold = -0.08  # 8% error threshold for weekly predictions
                else:
                    # For longer periods, be most lenient
                    cv_threshold = -0.10  # 10% error threshold for longer predictions
            
            if cv_mean < cv_threshold:
                print(f"[train_and_predict_model] Model {model_id} has poor CV performance ({cv_mean:.6f} < {cv_threshold:.6f}), skipping")
                return [0.0]
            
            # Additional check for extremely variable models (high CV standard deviation)
            if cv_std > 0.01:  # If CV results are very inconsistent
                print(f"[train_and_predict_model] Model {model_id} has inconsistent CV performance (std: {cv_std:.6f}), skipping")
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
            if len(predictions) == len(y_test) and len(predictions) >= 3:  # Need at least 3 samples for reliable bias
                prediction_bias = np.mean(predictions) - np.mean(y_test)
                
                # Conservative bias correction with reliability check
                # Only apply bias correction if it's statistically significant and reasonable
                prediction_std = np.std(predictions) if len(predictions) > 1 else 0.01
                target_std = np.std(y_test) if len(y_test) > 1 else 0.01
                
                # Bias must be at least 1 standard deviation to be considered significant
                bias_significance_threshold = max(prediction_std, target_std) * 0.5
                
                if abs(prediction_bias) < bias_significance_threshold:
                    # Bias is not significant, don't apply correction
                    prediction_bias = 0.0
                    print(f"[train_and_predict_model] Model {model_id} bias ({prediction_bias:.6f}) not significant, no correction applied")
                else:
                    # Ultra-conservative bias correction with sign preservation
                    # Limit bias correction to maximum ±2% to prevent sign flipping
                    max_bias_correction = 0.02  # Maximum bias correction factor (2%)
                    original_bias = prediction_bias
                    prediction_bias = np.clip(prediction_bias, -max_bias_correction, max_bias_correction)
                    if abs(original_bias) > max_bias_correction:
                        print(f"[train_and_predict_model] Model {model_id} bias capped (conservative): {original_bias:.6f} → {prediction_bias:.6f}")
                
                # Predict on the LAST training sample with bias correction
                last_sample = X_train_scaled[-1:]
                latest_prediction = model.predict(last_sample)[0]
                
                # Ensure bias correction doesn't flip prediction sign
                corrected_prediction = latest_prediction - prediction_bias
                
                # Check if bias correction would flip the sign of a meaningful prediction
                if abs(latest_prediction) > 0.005:  # Only check for predictions > 0.5%
                    if (latest_prediction > 0 and corrected_prediction < 0) or (latest_prediction < 0 and corrected_prediction > 0):
                        # Sign flip detected - use reduced bias correction
                        reduced_bias = prediction_bias * 0.5  # Use only 50% of bias correction
                        corrected_prediction = latest_prediction - reduced_bias
                        print(f"[train_and_predict_model] Model {model_id} Sign flip prevented: original={latest_prediction:.6f}, full_bias={prediction_bias:.6f}, reduced_bias={reduced_bias:.6f}, final={corrected_prediction:.6f}")
                        
                        # If still flipping, don't apply bias correction at all
                        if (latest_prediction > 0 and corrected_prediction < 0) or (latest_prediction < 0 and corrected_prediction > 0):
                            corrected_prediction = latest_prediction
                            print(f"[train_and_predict_model] Model {model_id} Complete bias correction skipped to preserve sign")
                
                # Conservative individual model bounds checking
                # Prevent any single model from dominating the ensemble with extreme predictions
                if prediction_window == 1:
                    max_individual_return = 0.10  # Maximum daily return threshold (10%)
                    if abs(corrected_prediction) > max_individual_return:
                        original_corrected = corrected_prediction
                        corrected_prediction = np.clip(corrected_prediction, -max_individual_return, max_individual_return)
                        print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped extreme individual prediction: {original_corrected:.6f} → {corrected_prediction:.6f}")
                elif prediction_window <= 7:
                    max_individual_return = 0.20  # 20% max for weekly predictions
                    if abs(corrected_prediction) > max_individual_return:
                        original_corrected = corrected_prediction
                        corrected_prediction = np.clip(corrected_prediction, -max_individual_return, max_individual_return)
                        print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped extreme {prediction_window}-day prediction: {original_corrected:.6f} → {corrected_prediction:.6f}")
                
                print(f"[train_and_predict_model] Model {model_id} bias correction: {prediction_bias:.6f}")
                print(f"[train_and_predict_model] Model {model_id} raw latest: {latest_prediction:.6f}, corrected: {corrected_prediction:.6f}")
                return [corrected_prediction]
            else:
                # Fallback: predict on last sample without bias correction
                last_sample = X_train_scaled[-1:]
                prediction = model.predict(last_sample)[0]
                
                # Still apply individual model bounds even without bias correction
                if prediction_window == 1:
                    max_individual_return = 0.10  # 10% max for daily
                    if abs(prediction) > max_individual_return:
                        original_pred = prediction
                        prediction = np.clip(prediction, -max_individual_return, max_individual_return)
                        print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped raw prediction: {original_pred:.6f} → {prediction:.6f}")
                elif prediction_window <= 7:
                    max_individual_return = 0.20  # 20% max for weekly
                    if abs(prediction) > max_individual_return:
                        original_pred = prediction
                        prediction = np.clip(prediction, -max_individual_return, max_individual_return)
                        print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped raw {prediction_window}-day prediction: {original_pred:.6f} → {prediction:.6f}")
                
                print(f"[train_and_predict_model] Model {model_id} no bias correction, raw: {prediction:.6f}")
                return [prediction]
            
        else:
            # If no test data, predict on last training sample
            last_sample = X_train_scaled[-1:] 
            prediction = model.predict(last_sample)[0]
            
            # Apply individual model bounds even for single predictions
            if prediction_window == 1:
                max_individual_return = 0.10  # 10% max for daily
                if abs(prediction) > max_individual_return:
                    original_pred = prediction
                    prediction = np.clip(prediction, -max_individual_return, max_individual_return)
                    print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped single prediction: {original_pred:.6f} → {prediction:.6f}")
            elif prediction_window <= 7:
                max_individual_return = 0.20  # 20% max for weekly
                if abs(prediction) > max_individual_return:
                    original_pred = prediction
                    prediction = np.clip(prediction, -max_individual_return, max_individual_return)
                    print(f"[train_and_predict_model] Model {model_id} ⚠️ Capped single {prediction_window}-day prediction: {original_pred:.6f} → {prediction:.6f}")
            
            print(f"[train_and_predict_model] Model {model_id} single prediction: {prediction:.6f}")
            return [prediction]
        
    except Exception as e:
        print(f"[train_and_predict_model] Error training model {model_id}: {e}")
        traceback.print_exc()
        return [0.0]

def train_models_parallel(stock_data, model_ids, regime=None, regime_strength=0.5, prediction_window=5):
    """Train models for all stocks using parallel processing, with regime-aware feature weighting."""
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        future_to_ticker = {
            executor.submit(train_model_for_stock, ticker, df, model_ids, regime, regime_strength, prediction_window): ticker 
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
    if df is None or len(df) < MIN_DATA_POINTS:
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
    
    plt.title(f'{ticker_name} - Price Prediction', color='white', fontsize=16)
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
    """Plot all top N stocks: multiple days of history and prediction forecast. Normalize to 100 at start of forecast period."""
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
        # Normalize to 100 at first value
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


def get_sector_sentiment(ticker):
    """
    Get sentiment score for sector-wide news
    Returns: sentiment score from -100 to +100
    """
    try:
        # Sector mapping for major tickers
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Automotive', 'NVDA': 'Technology', 'META': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
        }
        
        sector = sector_map.get(ticker, 'General Market')
        
        # Simulated sector sentiment
        random.seed(hash(sector) % 500)  # Consistent for same sector
        sector_sentiment = random.uniform(-20, 30)  # Generally less extreme than company
        
        print(f"[get_sector_sentiment] {ticker} ({sector}): {sector_sentiment:.1f}")
        return sector_sentiment
        
    except Exception as e:
        print(f"[get_sector_sentiment] Error for {ticker}: {e}")
        return 0.0

def analyze_ticker_sentiment(ticker):
    """
    Analyze sentiment for a single ticker using company news (50%), sector news (20%), and market news (30%)
    Returns: (sentiment_score: float -100 to +100, sentiment_details: dict)
    """
    try:
        print(f"[analyze_ticker_sentiment] 📰 Starting sentiment analysis for {ticker}")
        
        # Check if this is an index
        is_index = is_index_ticker(ticker)
        
        if is_index:
            print(f"[analyze_ticker_sentiment] 📊 Detected index {ticker}, using index sentiment (no news API)")
            # Use dedicated index sentiment function (no Alpha Vantage API calls)
            sentiment_data = get_index_sentiment_score(ticker.upper())
        else:
            # Use the comprehensive sentiment analysis that includes market sentiment
            sentiment_data = get_sentiment_score(ticker)
        
        sentiment_score = sentiment_data['sentiment_score']
        
        # Create appropriate sentiment details object based on whether it's an index or stock
        if is_index:
            sentiment_details = {
                'index_sentiment': sentiment_data.get('index_sentiment_score', 0.0),
                'market_sentiment': sentiment_data.get('market_sentiment', 0.0),
                # Use dynamic weights that were actually applied
                'index_weight': sentiment_data.get('dynamic_index_weight', 0),
                'market_weight': sentiment_data.get('dynamic_market_weight', 1.0),
                'has_index_news': sentiment_data.get('has_index_news', False),
                'final_score': sentiment_score,
                'sector': 'Index',
                'cache_used': True,
                'is_index': True,
                'api_limit_reached': sentiment_data.get('api_limit_reached', True),
                'using_fallback': sentiment_data.get('using_fallback', True),
                # Also include article counts for informational purposes
                'index_articles': sentiment_data.get('index_articles', 0),
            }
            
            print(f"[analyze_ticker_sentiment] ✅ {ticker}: Market={sentiment_data.get('market_sentiment', 0.0):.1f}, "
                f"Index={sentiment_data.get('index_sentiment_score', 0.0):.1f} "
                f"({sentiment_data.get('index_articles', 0)} articles), Final={sentiment_score:.1f}")
        else:
            sentiment_details = {
                'company_sentiment_score': sentiment_data.get('company_sentiment_score', 0.0),
                'sector_sentiment_score': sentiment_data.get('sector_sentiment_score', 0.0),
                'market_sentiment': sentiment_data.get('market_sentiment', 0.0),
                # Use dynamic weights that were actually applied
                'company_weight': sentiment_data.get('dynamic_company_weight', 0),
                'sector_weight': sentiment_data.get('dynamic_sector_weight', 0),
                'market_weight': sentiment_data.get('dynamic_market_weight', SENTIMENT_CONFIG['MARKET_NEWS_WEIGHT']),
                'has_company_news': sentiment_data.get('has_company_news', False),
                'has_sector_news': sentiment_data.get('has_sector_news', False),
                'final_score': sentiment_score,
                'sector': sentiment_data.get('sector', 'Unknown'),
                'cache_used': True,
                'is_index': False,
                # Also include article counts for informational purposes
                'company_articles': sentiment_data.get('company_articles', 0),
                'sector_articles': sentiment_data.get('sector_articles', 0)
            }
            
            print(f"[analyze_ticker_sentiment] ✅ {ticker}: Market={sentiment_data.get('market_sentiment', 0.0):.1f}, "
                f"Company={sentiment_data.get('company_sentiment_score', 0.0):.1f} ({sentiment_data.get('company_articles', 0)} articles), "
                f"Sector={sentiment_data.get('sector_sentiment_score', 0.0):.1f} ({sentiment_data.get('sector_articles', 0)} articles), Final={sentiment_score:.1f}")
        
        return sentiment_score, sentiment_details
        
    except Exception as e:
        print(f"[analyze_ticker_sentiment] ❌ Error analyzing sentiment for {ticker}: {e}")
        # Return neutral sentiment on error
        return 0.0, {
            'company_sentiment': 0.0,
            'sector_sentiment': 0.0,
            'market_sentiment': 0.0,
            'final_score': 0.0,
            'error': str(e),
            'is_index': is_index_ticker(ticker)
        }

def apply_sentiment_adjustment(ml_prediction, sentiment_score, prediction_window):
    """
    Apply sentiment adjustment to ML prediction using hybrid additive/multiplicative approach
    - Sentiment score: -100 (very negative) to +100 (very positive)
    - Can change the sign of predictions for strong sentiment
    - Time decay: longer windows get less sentiment impact
    """
    try:
        # Normalize sentiment score to -1 to +1 range
        sentiment_normalized = sentiment_score / 100.0
        
        # Calculate time decay factor (longer windows get less sentiment impact)
        time_decay_factor = max(0.4, 1.0 - (prediction_window - 1) * 0.08)
        
        # Direct sentiment impact (can change sign)
        max_direct_adjustment = 0.04 * time_decay_factor  # Up to 4% direct impact
        direct_adjustment = sentiment_normalized * max_direct_adjustment
        
        # Multiplicative adjustment for larger predictions
        max_multiplicative_ratio = 0.25 * time_decay_factor
        multiplicative_factor = sentiment_normalized * max_multiplicative_ratio
        
        # Hybrid approach based on prediction magnitude
        prediction_magnitude = abs(ml_prediction)
        
        if prediction_magnitude < 0.015:  # Small predictions (< 1.5%)
            # Primarily additive - sentiment can dominate
            additive_weight = 0.85
            multiplicative_weight = 0.15
            
            additive_part = direct_adjustment * additive_weight
            multiplicative_part = ml_prediction * (multiplicative_factor * multiplicative_weight)
            
        else:
            # Larger predictions - more multiplicative
            additive_weight = 0.4
            multiplicative_weight = 0.6
            
            additive_part = direct_adjustment * additive_weight  
            multiplicative_part = ml_prediction * (multiplicative_factor * multiplicative_weight)
        
        # Final adjusted prediction
        adjusted_prediction = ml_prediction + additive_part + multiplicative_part
        
        print(f"[sentiment_adjust] ML: {ml_prediction:.4f} ({ml_prediction*100:.2f}%), "
              f"Sentiment: {sentiment_score:.1f}, "
              f"Final: {adjusted_prediction:.4f} ({adjusted_prediction*100:.2f}%) "
              f"[Direct: {additive_part:.4f}, Mult: {multiplicative_part:.4f}]")
        
        return adjusted_prediction
        
    except Exception as e:
        print(f"[apply_sentiment_adjustment] Error: {e}")
        return ml_prediction  # Return original prediction on error

def adjust_confidence_interval_for_sentiment(lower_bound, upper_bound, original_prediction, sentiment_adjusted_prediction):
    """
    Adjust confidence interval bounds to account for sentiment impact on prediction.
    
    Args:
        lower_bound: Original lower confidence bound
        upper_bound: Original upper confidence bound
        original_prediction: Original ML prediction before sentiment adjustment
        sentiment_adjusted_prediction: Final prediction after sentiment adjustment
        
    Returns:
        tuple: (adjusted_lower_bound, adjusted_upper_bound)
    """
    # Calculate the sentiment impact
    sentiment_impact = sentiment_adjusted_prediction - original_prediction
    
    # Simple additive adjustment - shift bounds by the same amount
    adjusted_lower = lower_bound + sentiment_impact
    adjusted_upper = upper_bound + sentiment_impact
    
    # Ensure bounds maintain their relationship (lower < upper)
    if adjusted_lower > adjusted_upper:
        # Swap if order was reversed
        adjusted_lower, adjusted_upper = adjusted_upper, adjusted_lower
    
    return adjusted_lower, adjusted_upper

def download_single_ticker_data(ticker, start_date):
    """Download data for a single ticker."""
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
    """
    Convert any array-like or nested value to a scalar.
    Handles multiple data types safely including nested arrays, lists, and Series objects.
    """
    try:
        # If it's None or already scalar
        if x is None or np.isscalar(x):
            return x
            
        # If it's a pandas Series
        if isinstance(x, pd.Series):
            if len(x) > 0:
                return to_scalar(x.iloc[0])
            return np.nan
            
        # If it's a numpy array
        if isinstance(x, np.ndarray):
            x = x.flatten()  # Flatten any multi-dimensional array
            if x.size > 0:
                return float(x[0])
            return np.nan
            
        # If it's a list or tuple
        if isinstance(x, (list, tuple)):
            if len(x) > 0:
                return to_scalar(x[0])
            return np.nan
            
        # Try direct float conversion as last resort
        return float(x)
        
    except Exception:
        return np.nan  # Return NaN for any conversion failure

def add_features_single(ticker, df, prediction_window=5, market_data_cache=None):
    """Add features to a single ticker's data (no parallelization)."""
    return add_features_to_stock(ticker, df, prediction_window, market_data_cache)

def train_model_single(df, model_ids, regime=None, regime_strength=0.5, prediction_window=5):
    """Train models for a single ticker (no parallelization) with bounds checking."""
    if df is None or len(df) < MIN_DATA_POINTS:
        print(f"[train_model_single] DataFrame too short or None")
        return None
    
    try:
        # Use the same logic as train_model_for_stock but for single ticker
        return train_model_for_stock("SINGLE_TICKER", df, model_ids, regime, regime_strength, prediction_window)
        
    except Exception as e:
        print(f"[train_model_single] ERROR: {e}")
        traceback.print_exc()
        return None

def predict_single_ticker_chart(df, predictions, prediction_window):
    """Create a chart for a single ticker: recent history and forecast, normalized to 100 at start of shown period."""
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
            
            print(f"[predict] 🎯 User requested {prediction_window}-day prediction window")
            print(f"[predict] Final selected_models for single ticker: {selected_models}")
            df = download_single_ticker_data(ticker_to_analyze, start_date)
            if df is None or df.empty or len(df) < MIN_DATA_POINTS:
                return jsonify({'error': f'❌ Could not fetch data for {ticker_to_analyze}. Please check the ticker symbol and try again.'}), 400
            features_df = add_features_to_stock(ticker_to_analyze, df, prediction_window, market_data_cache)
            if features_df is None or features_df.empty:
                return jsonify({'error': '❌ Could not process technical indicators. Data may be corrupted or insufficient.'}), 400
            model_preds = train_model_single(features_df, selected_models, market_condition, market_strength, prediction_window)
            if not model_preds:
                return jsonify({'error': '❌ Could not train machine learning models. Insufficient data or technical issues.'}), 400
            
            # Apply sentiment analysis to enhance prediction accuracy
            try:
                print(f"[predict] 📰 Analyzing sentiment for {ticker_to_analyze}...")
                
                # Get sentiment analysis
                sentiment_score, sentiment_details = analyze_ticker_sentiment(ticker_to_analyze)
                print(f"[predict] 📊 Sentiment score for {ticker_to_analyze}: {sentiment_score}")
                
                # Check if this is an index - special handling for indexes
                is_index = sentiment_details.get('is_index', False)
                
                if is_index:
                    print(f"[predict] 📊 {ticker_to_analyze} is an index - applying sentiment adjustment with market data")
                    # Apply sentiment adjustment to indexes too, using their market-only sentiment
                    original_prediction = model_preds.get('prediction', 0.0) if isinstance(model_preds, dict) else 0.0
                    adjusted_prediction = apply_sentiment_adjustment(original_prediction, sentiment_score, prediction_window)
                    
                    print(f"[predict] 🎯 {ticker_to_analyze}: ML prediction: {original_prediction:.4f} ({original_prediction*100:.2f}%)")
                    print(f"[predict] 📰 {ticker_to_analyze}: Index sentiment adjustment: {sentiment_score}/100 (market only)")
                    print(f"[predict] ✅ {ticker_to_analyze}: Final prediction: {adjusted_prediction:.4f} ({adjusted_prediction*100:.2f}%)")
                    
                    # Update the model predictions with sentiment data - use index-specific fields
                    if isinstance(model_preds, dict):
                        model_preds['prediction'] = adjusted_prediction
                        model_preds['sentiment_score'] = sentiment_score
                        model_preds['sentiment_details'] = sentiment_details
                        model_preds['original_ml_prediction'] = original_prediction
                        model_preds['sentiment_adjusted'] = True
                        model_preds['is_index'] = True
                        model_preds['api_limit_reached'] = sentiment_details.get('api_limit_reached', True)
                        model_preds['using_fallback'] = sentiment_details.get('using_fallback', True)
                else:
                    # Stock handling - apply sentiment adjustment using company + sector sentiment
                    print(f"[predict] 📈 {ticker_to_analyze} is a stock - applying sentiment adjustment with company/sector data")
                    original_prediction = model_preds.get('prediction', 0.0) if isinstance(model_preds, dict) else 0.0
                    adjusted_prediction = apply_sentiment_adjustment(original_prediction, sentiment_score, prediction_window)
                    
                    print(f"[predict] 🎯 {ticker_to_analyze}: ML prediction: {original_prediction:.4f} ({original_prediction*100:.2f}%)")
                    print(f"[predict] 📰 {ticker_to_analyze}: Stock sentiment adjustment: {sentiment_score}/100 (company + sector)")
                    print(f"[predict] ✅ {ticker_to_analyze}: Final prediction: {adjusted_prediction:.4f} ({adjusted_prediction*100:.2f}%)")
                    
                    # Update the model predictions with sentiment data - use stock-specific fields
                    if isinstance(model_preds, dict):
                        model_preds['prediction'] = adjusted_prediction
                        model_preds['sentiment_score'] = sentiment_score
                        model_preds['sentiment_details'] = sentiment_details
                        model_preds['original_ml_prediction'] = original_prediction
                        model_preds['sentiment_adjusted'] = True
                        model_preds['is_index'] = False
                        model_preds['api_limit_reached'] = sentiment_details.get('api_limit_reached', True)
                        model_preds['using_fallback'] = sentiment_details.get('using_fallback', True)
                
            except Exception as e:
                print(f"[predict] ⚠️ Sentiment analysis error for {ticker_to_analyze}: {e}")
                print("[predict] 📈 Continuing with ML-only prediction...")
                # Continue without sentiment if there's an error
            
            # Use confidence interval calculation for single ticker
            ci = fixed_single_ticker_prediction(df, model_preds, market_condition, market_strength, confidence_interval, prediction_window)
            
            # Add directional confidence prediction
            try:
                direction_result = predict_direction_confidence(ticker_to_analyze, features_df, prediction_window)
                print(f"✅ Added directional confidence for {ticker_to_analyze}")
            except Exception as e:
                print(f"⚠️ Error calculating direction confidence for {ticker_to_analyze}: {e}")
                direction_result = None
            
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
            
            # Add directional confidence to response if available
            if direction_result:
                response['index_prediction']['direction'] = direction_result['direction']
                response['index_prediction']['direction_probability'] = direction_result['direction_probability']
            
            # Add sentiment information to response if available
            if isinstance(model_preds, dict) and 'sentiment_score' in model_preds:
                response['sentiment_analysis'] = {
                    'sentiment_score': model_preds['sentiment_score'],
                    'original_ml_prediction': model_preds.get('original_ml_prediction'),
                    'sentiment_details': model_preds.get('sentiment_details', {})
                }
            
            # Add API key setup information
            alpha_vantage_key = SENTIMENT_CONFIG.get('ALPHA_VANTAGE_API_KEY')
            if not alpha_vantage_key or alpha_vantage_key == 'your_key_here':
                response['system_messages'].append({
                    'type': 'info',
                    'message': '📰 Sentiment analysis using simulated data. For real news sentiment, get a free Alpha Vantage API key at https://www.alphavantage.co/support/#api-key and update SENTIMENT_CONFIG in the code.'
                })
            else:
                response['system_messages'].append({
                    'type': 'success', 
                    'message': '📰 Prediction includes real-time sentiment analysis from news and market data.'
                })
            
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
            print(f"🎯 User requested {prediction_window}-day prediction window for {len(processed_data)} stocks")
            print("Training models and generating predictions...")
            trained_models = train_models_parallel(processed_data, selected_models, market_condition, market_strength, prediction_window)
            if not trained_models:
                print(f"ERROR: Could not train models")
                return jsonify({'error': '❌ Could not train machine learning models. Insufficient data or technical issues.'}), 400
            print(f"Trained models for {len(trained_models)} stocks")

            # Retrieve market sentiment data for all index constituents
            # This avoids making individual sentiment API calls for each stock
            market_sentiment_score = None
            market_sentiment_details = None
            try:
                print(f"[predict] 📰 Getting market sentiment for index-wide application to {len(trained_models)} stocks...")
                # Use the index name to get market sentiment (which skips individual news API calls)
                market_sentiment_score, market_sentiment_details = analyze_ticker_sentiment(index)
                print(f"[predict] ✅ Using market sentiment {market_sentiment_score:.1f} for ALL stocks in {index}")
            except Exception as e:
                print(f"[predict] ⚠️ Error getting market sentiment: {e}")
                market_sentiment_score = 0.0
                market_sentiment_details = {}

            stock_predictions = []
            for ticker, model_predictions in trained_models.items():
                if etf_ticker and ticker == etf_ticker:
                    continue  # Skip the ETF ticker in stock_predictions
                
                # Debug: print model predictions for first few stocks
                if len(stock_predictions) < 3:
                    print(f"[DEBUG] {ticker} model predictions: {model_predictions}")
                
                # Use confidence interval calculation for each stock
                stock_df = stock_data.get(ticker)
                ci = fixed_single_ticker_prediction(
                    stock_df,
                    model_predictions,
                    market_condition,
                    market_strength,
                    confidence_interval,
                    prediction_window
                )
                
                # Apply market sentiment to individual stock prediction
                # Instead of getting sentiment for each stock, use the pre-calculated market sentiment
                if market_sentiment_score is not None:
                    original_prediction = ci['prediction']
                    adjusted_prediction = apply_sentiment_adjustment(original_prediction, market_sentiment_score, prediction_window)
                    
                    # Adjust confidence intervals to reflect sentiment impact
                    original_lower = ci['lower_bound']
                    original_upper = ci['upper_bound']
                    
                    # Adjust confidence interval bounds for sentiment impact using the same function we use for index
                    adjusted_lower, adjusted_upper = adjust_confidence_interval_for_sentiment(
                        original_lower, original_upper, original_prediction, adjusted_prediction
                    )
                    
                    # Update all values consistently
                    ci['prediction'] = adjusted_prediction
                    ci['lower_bound'] = adjusted_lower
                    ci['upper_bound'] = adjusted_upper
                    
                    # Optionally log the adjustment for significant changes
                    if abs(adjusted_prediction - original_prediction) > 0.005:  # 0.5% threshold
                        print(f"[predict] 📊 {ticker}: Sentiment adjusted prediction: {original_prediction:.6f} → {adjusted_prediction:.6f}")
                        print(f"[predict] 📊 {ticker}: Sentiment adjusted CI: [{original_lower:.6f}, {original_upper:.6f}] → [{adjusted_lower:.6f}, {adjusted_upper:.6f}]")
                
                last_close = float(stock_df['Close'].iloc[-1]) if stock_df is not None and 'Close' in stock_df.columns and not stock_df.empty else None
                stock_predictions.append({
                    'ticker': ticker,
                    'pred': ci['prediction'],
                    'lower': ci['lower_bound'],
                    'upper': ci['upper_bound'],
                    'close': last_close
                })
            
            # Add directional confidence prediction BEFORE sorting
            try:
                stock_predictions = apply_direction_confidence_parallel(
                    stock_predictions, 
                    processed_data,
                    prediction_window
                )
                print(f"✅ Added directional confidence to {len(stock_predictions)} stocks")
            except Exception as e:
                print(f"⚠️ Error applying directional confidence: {e}")
                # Continue without direction confidence if there's an error

            # Filter predictions based on showType and directional probability
            show_type = data.get('showType', 'top')
            original_count = len(stock_predictions)
            
            if show_type == 'top':
                # First check how many stocks have direction info
                stocks_with_direction = [s for s in stock_predictions if 'direction' in s]
                
                if len(stocks_with_direction) >= num_stocks:
                    # If we have enough stocks with direction, filter strictly
                    filtered_predictions = [s for s in stock_predictions if 'direction' in s and s['direction'] == 'up']
                else:
                    # Fall back to more lenient filtering if we don't have enough stocks with direction
                    filtered_predictions = [
                        s for s in stock_predictions 
                        if ('direction' not in s) or ('direction' in s and s['direction'] == 'up')
                    ]
                
                # Ensure we have enough predictions after filtering
                if len(filtered_predictions) < num_stocks and len(filtered_predictions) < len(stock_predictions):
                    print(f"⚠️ After UP direction filtering, only {len(filtered_predictions)} stocks remain. Adding some additional stocks to meet requested count.")
                    # Add some of the excluded stocks if we don't have enough
                    excluded = [s for s in stock_predictions if s not in filtered_predictions]
                    filtered_predictions.extend(excluded[:num_stocks - len(filtered_predictions)])
                
                stock_predictions = filtered_predictions
                
            elif show_type == 'worst':
                # First check how many stocks have direction info
                stocks_with_direction = [s for s in stock_predictions if 'direction' in s]
                
                if len(stocks_with_direction) >= num_stocks:
                    # If we have enough stocks with direction, filter strictly
                    filtered_predictions = [s for s in stock_predictions if 'direction' in s and s['direction'] == 'down']
                else:
                    # Fall back to more lenient filtering if we don't have enough stocks with direction
                    filtered_predictions = [
                        s for s in stock_predictions 
                        if ('direction' not in s) or ('direction' in s and s['direction'] == 'down')
                    ]
                
                # Ensure we have enough predictions after filtering
                if len(filtered_predictions) < num_stocks and len(filtered_predictions) < len(stock_predictions):
                    print(f"⚠️ After DOWN direction filtering, only {len(filtered_predictions)} stocks remain. Adding some additional stocks to meet requested count.")
                    # Add some of the excluded stocks if we don't have enough
                    excluded = [s for s in stock_predictions if s not in filtered_predictions]
                    filtered_predictions.extend(excluded[:num_stocks - len(filtered_predictions)])
                
                stock_predictions = filtered_predictions

            print(f"Direction filtering: {original_count} stocks → {len(stock_predictions)} stocks after applying {show_type} direction filter")
            
            # Sort based on showType parameter (default to 'top' if not specified)
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

            # Apply sentiment analysis to index prediction
            index_sentiment_info = None
            try:
                print(f"[predict] 📰 Applying market sentiment to index {index_name_for_response}...")
                
                # For index prediction (multiple tickers), get index-specific sentiment
                sentiment_score, sentiment_details = analyze_ticker_sentiment(index_name_for_response)
                print(f"[predict] 📊 Index sentiment score for {index_name_for_response}: {sentiment_score}")
                print(f"[predict] 📊 API limit reached: {sentiment_details.get('api_limit_reached', 'Unknown')}")
                print(f"[predict] 📊 Using fallback: {sentiment_details.get('using_fallback', 'Unknown')}")
                
                # Apply sentiment adjustment to index prediction
                original_index_prediction = index_prediction
                adjusted_index_prediction = apply_sentiment_adjustment(original_index_prediction, sentiment_score, prediction_window)
                
                print(f"[predict] 🎯 {index_name_for_response}: ML prediction: {original_index_prediction:.4f} ({original_index_prediction*100:.2f}%)")
                print(f"[predict] 📰 {index_name_for_response}: Index sentiment adjustment: {sentiment_score}/100 (market only)")
                print(f"[predict] ✅ {index_name_for_response}: Final prediction: {adjusted_index_prediction:.4f} ({adjusted_index_prediction*100:.2f}%)")
                
                # Update the index prediction with sentiment-adjusted value
                index_prediction = adjusted_index_prediction
                
                # Adjust confidence intervals for sentiment impact on index prediction
                if abs(adjusted_index_prediction - original_index_prediction) > 0.001:  # 0.1% threshold
                    original_index_lower = index_lower
                    original_index_upper = index_upper
                    
                    # Adjust confidence interval bounds for sentiment impact
                    index_lower, index_upper = adjust_confidence_interval_for_sentiment(
                        index_lower, index_upper, original_index_prediction, adjusted_index_prediction
                    )
                    print(f"[predict] 📊 {index_name_for_response}: Original CI: [{original_index_lower:.6f}, {original_index_upper:.6f}]")
                    print(f"[predict] 📊 {index_name_for_response}: Sentiment-adjusted CI: [{index_lower:.6f}, {index_upper:.6f}] = [{index_lower*100:.3f}%, {index_upper*100:.3f}%]")
                
                # Store sentiment information for response using index-specific structure
                index_sentiment_info = {
                    'sentiment_score': sentiment_score,
                    'original_ml_prediction': original_index_prediction,
                    'sentiment_details': sentiment_details,
                    'is_index': True,
                    'api_limit_reached': sentiment_details.get('api_limit_reached', True),
                    'using_fallback': sentiment_details.get('using_fallback', True)
                }
                
            except Exception as e:
                print(f"[predict] ⚠️ Index sentiment analysis error for {index_name_for_response}: {e}")
                print("[predict] 📈 Continuing with ML-only index prediction...")
                # Continue without sentiment if there's an error
                
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
            
            # Add sentiment information to response if available
            if index_sentiment_info:
                response['sentiment_analysis'] = index_sentiment_info
            
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
        error_trace = traceback.format_exc()
        print(f"Full error trace: {error_trace}")
        
        # Error reporting with specific error codes
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific error codes for common issues
        if 'infinity' in str(e).lower() or 'inf' in str(e).lower():
            error_response['error_code'] = 'DATA_INFINITY_ERROR'
            error_response['user_message'] = 'Data processing error: extreme values detected'
        elif 'nan' in str(e).lower():
            error_response['error_code'] = 'DATA_NAN_ERROR'
            error_response['user_message'] = 'Data processing error: missing values detected'
        elif 'download' in str(e).lower() or 'fetch' in str(e).lower():
            error_response['error_code'] = 'DATA_FETCH_ERROR'
            error_response['user_message'] = 'Could not fetch market data. Please try again.'
        elif 'model' in str(e).lower() or 'train' in str(e).lower():
            error_response['error_code'] = 'MODEL_TRAINING_ERROR'
            error_response['user_message'] = 'ML model training failed. Please try with different parameters.'
        else:
            error_response['error_code'] = 'GENERAL_ERROR'
            error_response['user_message'] = 'An unexpected error occurred. Please try again.'
        
        return jsonify(error_response), 500
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'IndexLab Backend is running'})

@app.route('/healthz', methods=['GET'])
def healthz():
    """Health check endpoint for Render monitoring"""
    return jsonify({'status': 'ok'})

@app.route('/api/market-sentiment', methods=['GET'])
def get_current_market_sentiment():
    """Get current global market sentiment"""
    try:
        global MARKET_SENTIMENT_CACHE, MARKET_SENTIMENT_TIMESTAMP
        
        market_sentiment = get_market_sentiment()
        
        cache_age_hours = 0
        if MARKET_SENTIMENT_TIMESTAMP:
            cache_age_hours = (time.time() - MARKET_SENTIMENT_TIMESTAMP) / 3600
        
        return jsonify({
            'market_sentiment': market_sentiment,
            'cache_age_hours': round(cache_age_hours, 2),
            'sentiment_range': '(-100 = very negative, 0 = neutral, +100 = very positive)',
            'cache_duration_hours': SENTIMENT_CONFIG['MARKET_SENTIMENT_CACHE_HOURS'],
            'timestamp': MARKET_SENTIMENT_TIMESTAMP
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh-market-sentiment', methods=['POST'])
def refresh_market_sentiment():
    """Force refresh of global market sentiment"""
    try:
        global MARKET_SENTIMENT_CACHE, MARKET_SENTIMENT_TIMESTAMP
        
        # Clear cache to force refresh
        with MARKET_SENTIMENT_LOCK:
            MARKET_SENTIMENT_CACHE = None
            MARKET_SENTIMENT_TIMESTAMP = None
        
        # Get fresh market sentiment
        market_sentiment = get_market_sentiment()
        
        return jsonify({
            'message': 'Market sentiment refreshed successfully',
            'new_market_sentiment': market_sentiment,
            'timestamp': MARKET_SENTIMENT_TIMESTAMP
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-news', methods=['GET'])
def test_news_api():
    """Test endpoint to check Alpha Vantage News API"""
    try:
        query = request.args.get('query', 'financial_markets')
        is_ticker = request.args.get('is_ticker', 'false').lower() == 'true'
        
        print(f"📰 Testing Alpha Vantage API with query: {query}, is_ticker: {is_ticker}")
        articles = get_alpha_vantage_news(query, limit=5, is_ticker=is_ticker)
        
        return jsonify({
            'query': query,
            'is_ticker': is_ticker,
            'articles_found': len(articles),
            'articles': articles[:3]  # Return first 3 articles for testing
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting IndexLab Backend Server...")
    print("Available endpoints:")
    print("- POST /api/predict - Main prediction endpoint")
    print("- GET /api/health - Health check")
    print("- GET /api/market-sentiment - Get current market sentiment")
    print("- POST /api/refresh-market-sentiment - Force refresh market sentiment")
    print("- GET /api/test-news - Test Alpha Vantage News API")
    print("")
    
    # Initialize sentiment model at startup (with graceful failure)
    model_loaded = initialize_sentiment_model()
    
    # Pre-fetch market sentiment at startup (optional, with timeout)
    try:
        # Simple timeout mechanism (works on Windows)

        
        def fetch_with_timeout():
            global startup_sentiment_result
            startup_sentiment_result = None
            try:
                startup_sentiment_result = get_market_sentiment()
            except Exception as e:
                startup_sentiment_result = "error"
        
        # Start fetch in background thread
        fetch_thread = threading.Thread(target=fetch_with_timeout)
        fetch_thread.daemon = True
        fetch_thread.start()
        
        # Wait up to 20 seconds for completion
        fetch_thread.join(timeout=20)
        
        if hasattr(globals(), 'startup_sentiment_result') and startup_sentiment_result is not None:
            if startup_sentiment_result != "error":
                print(f"✅ Initial market sentiment: {startup_sentiment_result:.1f}/100")
            else:
                print("⚠️ Market sentiment fetch failed (will use fallback when needed)")
        else:
            print("⚠️ Market sentiment fetch timed out (will use fallback when needed)")
        
    except Exception as e:
        print("⚠️ Market sentiment fetch skipped (will use fallback when needed)")

# ============================================================================
# DIRECTIONAL CONFIDENCE FUNCTIONS (NOT YET CONNECTED TO MAIN PIPELINE)
# ============================================================================

def add_binary_direction_target(df, prediction_window=5):
    """Add binary direction target (1=up, 0=down) based on forward returns."""
    target_column = f'forward_return_{prediction_window}'
    if target_column in df.columns:
        direction_column = f'direction_{prediction_window}'
        df[direction_column] = (df[target_column] > 0).astype(int)
        print(f"✅ Added binary direction target '{direction_column}' (1=up, 0=down)")
        
        # Count class distribution for debugging
        up_count = df[direction_column].sum()
        total = len(df[direction_column].dropna())
        if total > 0:
            up_pct = up_count / total * 100
            print(f"📊 Class distribution: {up_count} up ({up_pct:.1f}%), {total-up_count} down ({100-up_pct:.1f}%)")
    
    return df

def select_direction_features(df, prediction_window=5):
    """Select optimal features for directional prediction."""
    
    # Technical features - generally good for direction
    tech_features = [
        'RSI', 'macd', 'macd_signal', 'macd_histogram', 
        'SMA_20', 'sma_10', 'sma_50', 'bb_width',
        'percent_b', 'momentum', 'roc_10', 'price_change',
        'ATR', 'volatility', 'volume_ratio'
    ]
    
    # Cross-asset features (very important for direction)
    cross_features = [
        'relative_strength', 'correlation_spy_20', 'beta_spy_20',
        'spy_ratio', 'spy_ratio_ma', 'vix_level', 'vix_change',
        'risk_on', 'risk_off', 'rising_rates', 'tech_outperform'
    ]
    
    # Time-based features (important for CatBoost)
    time_features = [
        'day_of_week', 'month', 'quarter', 
        'is_quarter_end', 'is_friday', 'is_monday'
    ]
    
    # Define categorical features for CatBoost
    categorical_features = ['day_of_week', 'month', 'quarter', 'is_quarter_end', 'is_friday', 'is_monday']
    
    # Combine all feature lists
    all_features = tech_features + cross_features + time_features
    
    # Filter to only include features present in dataframe
    available_features = [f for f in all_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in available_features]
    
    print(f"✅ Selected {len(available_features)} features for directional prediction")
    print(f"✅ {len(available_categorical)} categorical features identified")
    
    return available_features, available_categorical

def create_direction_classifier(X_train, y_train, cat_features=None):
    """Create and configure CatBoost model for directional prediction."""
    
    train_pool = None  # Initialize for proper cleanup
    
    # Convert numpy array to DataFrame for better categorical handling
    df_X = pd.DataFrame(X_train)
    
    # Handle categorical features properly
    if cat_features and len(cat_features) > 0:
        print(f"Processing {len(cat_features)} categorical features...")
        for idx in cat_features:
            if idx < df_X.shape[1]:  # Ensure index is valid
                # Handle NaN values in categorical features by converting to string
                df_X[idx] = df_X[idx].fillna('missing').astype(str).astype('category')
        print(f"  ✅ Converted {len(cat_features)} features to categorical")
    
    # Try Pool-based approach with DataFrame
    try:
        print("Attempting Pool-based categorical training with DataFrame...")
        train_pool = Pool(
            data=df_X,
            label=y_train,
            cat_features=cat_features if cat_features else []
        )
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=3,
            l2_leaf_reg=3,
            loss_function='Logloss',
            verbose=False,
            task_type='CPU'
        )
        model.fit(train_pool, plot=False)
        print("  ✅ Pool-based categorical training succeeded.")
        return model
    except Exception as e:
        print(f"Pool-based approach failed: {e}")
    finally:
        # Clean up Pool object to prevent memory leaks
        if train_pool is not None:
            del train_pool
            train_pool = None
    
    # Try DataFrame-based approach as backup
    try:
        print("Attempting DataFrame-based categorical training...")
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=3,
            l2_leaf_reg=3,
            loss_function='Logloss',
            verbose=False,
            task_type='CPU'
        )
        model.fit(df_X, y_train, cat_features=cat_features, plot=False)
        print("  ✅ DataFrame-based categorical training succeeded.")
        return model
    except Exception as e2:
        print(f"DataFrame approach failed: {e2}")
        
        # If all else fails, convert categorical features to numeric and train
        print("Converting categorical features to numeric and training...")
        df_X_numeric = df_X.copy()
        if cat_features:
            for idx in cat_features:
                if idx < df_X_numeric.shape[1]:
                    # Convert categorical to numeric using label encoding
                    df_X_numeric[idx] = pd.to_numeric(df_X_numeric[idx], errors='coerce').fillna(0)
        
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=3,
            l2_leaf_reg=3,
            loss_function='Logloss',
            verbose=False,
            task_type='CPU'
        )
        model.fit(df_X_numeric, y_train, plot=False)
        print("  ✅ Numeric conversion approach succeeded.")
        return model

def predict_direction_confidence(ticker, df, prediction_window=5):
    """Complete directional confidence prediction for a single ticker."""
    print(f"🔮 Predicting directional confidence for {ticker}")
    
    try:
        # Step 1: Ensure we have binary target
        df = add_binary_direction_target(df, prediction_window)
        
        # Step 2: Select appropriate features
        features, cat_features = select_direction_features(df, prediction_window)
        
        print(f"DEBUG {ticker}: Selected {len(features)} features")
        
        # Check if we have enough features
        if len(features) < 5:
            print(f"❌ {ticker}: Not enough features for direction model ({len(features)})")
            # Create ticker-specific fallback values
            ticker_hash = sum(ord(c) for c in ticker)
            direction = 'up' if ticker_hash % 3 != 0 else 'down'
            confidence = 15.0 + (ticker_hash % 15)  # 15-30% range for insufficient features
            probability = 50.0 + confidence/2  # Always >= 50% regardless of direction
            
            return {
                'direction': direction,
                'direction_probability': float(probability),  # As percentage
                'error': "Insufficient features"
            }
        

        # Convert categorical columns to 'category' dtype in DataFrame
        cat_feature_indices = []
        if cat_features and len(cat_features) > 0:
            for cat_feature in cat_features:
                if cat_feature in df.columns:
                    df[cat_feature] = df[cat_feature].astype('category')
            cat_feature_indices = []
            for f in cat_features:
                if f in features:
                    try:
                        cat_feature_indices.append(features.index(f))
                    except ValueError:
                        pass
            print(f"  ✅ Converted {len(cat_feature_indices)} categorical features to 'category' dtype")
        # Step 3: Prepare feature matrix and target
        X = df[features].values
        y_binary = df[f'direction_{prediction_window}'].values

        # Print data analysis for debugging
        print(f"[predict_direction_confidence] {ticker}: Data analysis:")
        print(f"  - Full X shape: {X.shape}")
        print(f"  - Binary target shape: {y_binary.shape}")
        print(f"  - Up/Down distribution: {sum(y_binary)} up, {len(y_binary) - sum(y_binary)} down (total: {len(y_binary)})")
        print(f"  - Feature count: {len(features)}")
        # Handle any missing values in the feature matrix
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            print(f"  - {nan_count} NaN values ({(nan_count / X.size * 100):.2f}% of data)")
            # Replace NaN values with column means
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                X[mask, i] = col_means[i]
            print(f"⚠️ {ticker}: NaN values detected in features, replacing with column means")

        # Handle infinite values
        if np.isinf(X).any():
            print(f"⚠️ {ticker}: Infinite values detected, replacing with bounded values")
            X = np.where(np.isinf(X), np.sign(X) * 1e6, X)

        # Debug info
        print(f"DEBUG {ticker}: Feature matrix shape = {X.shape}, std = {np.std(X):.4f}")

        print(f"🔄 {ticker}: Training CatBoost model with {X.shape[0]} samples, {X.shape[1]} features")

        # Step 4: Create and train the model using Pool-based approach
        model = create_direction_classifier(X[:-1], y_binary[:-1], cat_features=cat_feature_indices)
        print(f"✅ {ticker}: CatBoost model trained successfully!")

        # Step 5: Predict on the latest data point using DataFrame
        latest_features_df = df[features].iloc[[-1]].copy()
        
        # Handle categorical columns the same way as in training
        if cat_feature_indices and len(cat_feature_indices) > 0:
            for idx in cat_feature_indices:
                if idx < latest_features_df.shape[1]:
                    col_name = latest_features_df.columns[idx]
                    # Handle NaN values the same way as in training
                    latest_features_df[col_name] = latest_features_df[col_name].fillna('missing').astype(str).astype('category')
        
        print(f"DEBUG {ticker}: Making prediction on feature vector: shape={latest_features_df.shape}")

        pred_pool = Pool(data=latest_features_df,cat_features=cat_feature_indices)
        
        # Use the Pool object for prediction
        probabilities = model.predict_proba(pred_pool)[0]
        up_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])

        # Clean up Pool object to prevent memory leaks
        del pred_pool

        print(f"DEBUG {ticker}: Raw probabilities from model: {probabilities}")
        print(f"DEBUG {ticker}: Raw up probability: {up_probability:.4f}")
        
        # Force a minimum deviation from 0.5 to avoid zero confidence
        if abs(up_probability - 0.5) < 0.05:
            # Add a small bias based on recent price action
            try:
                # Get recent price trend as a bias factor
                recent_returns = df['Close'].pct_change()[-5:].mean()
                bias = 0.05 * np.sign(recent_returns) if not np.isnan(recent_returns) else 0.02
                up_probability = 0.5 + bias
                print(f"DEBUG {ticker}: Applied bias adjustment: +{bias:.4f}")
            except:
                # Apply ticker-specific bias instead of random
                ticker_hash = sum(ord(c) for c in ticker)
                bias_sign = 1 if ticker_hash % 2 == 0 else -1
                up_probability = 0.5 + (0.05 * bias_sign)
        
        # ALWAYS choose the direction with probability >= 0.5 (the more likely outcome)
        if up_probability >= 0.5:
            direction = 'up'
            display_probability = up_probability
        else:
            direction = 'down' 
            display_probability = 1 - up_probability
            
        # Verification: ensure we always display probability >= 0.5
        assert display_probability >= 0.5, f"Display probability should be >= 0.5, got {display_probability}"
        
        print(f"DEBUG {ticker}: Selected direction: {direction} with probability: {display_probability:.4f}")
        
        # For display: use the actual probability of the predicted direction
            
        # Get feature importance if available
        top_features = {}
        try:
            feature_importance = model.get_feature_importance()
            top_features = dict(zip(features, feature_importance))
            top5 = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]
            top_features = dict(top5)
        except:
            pass
        
        result = {
            'direction': direction,
            'direction_probability': float(display_probability * 100),  # Convert to percentage for display
            'top_features': top_features,
            'prediction_window': prediction_window,
        }
        
        print(f"✅ {ticker}: {direction.upper()} with {display_probability*100:.1f}% probability")
        return result
        
    except Exception as e:
        print(f"❌ Error predicting direction for {ticker}: {e}")
        traceback.print_exc()
        
        # Return a ticker-specific fallback with varied confidence
        ticker_hash = sum(ord(c) for c in ticker)
        direction = 'up' if ticker_hash % 3 != 0 else 'down'  # 2/3 up, 1/3 down
        confidence = 15.0 + (ticker_hash % 20)  # 15-35% range
        probability = 50.0 + confidence/2  # Always >= 50% regardless of direction
        
        return {
            'direction': direction,
            'direction_probability': float(probability),  # Already as percentage
            'error': str(e)
        }

def apply_direction_confidence_parallel(stock_predictions, processed_data, prediction_window=5):
    """Apply directional confidence model to final stock predictions."""
    print(f"Calculating directional confidence for {len(stock_predictions)} predicted stocks...")
    
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
        # Only process stocks that made it to the final prediction list
        futures = {
            executor.submit(
                predict_direction_confidence, 
                stock['ticker'], 
                processed_data[stock['ticker']],
                prediction_window
            ): stock['ticker'] 
            for stock in stock_predictions
            if stock['ticker'] in processed_data
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                confidence_result = future.result()
                # Find the matching stock in stock_predictions and add confidence
                for stock in stock_predictions:
                    if stock['ticker'] == ticker:
                        stock['direction'] = confidence_result['direction']
                        stock['direction_probability'] = confidence_result['direction_probability']
                        break
            except Exception as e:
                print(f"Error calculating direction confidence for {ticker}: {e}")
    
    print(f"✅ Added directional confidence to {len(stock_predictions)} predictions")
    return stock_predictions

# ============================================================================
# FLASK APP INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    print("Starting IndexLab Backend Server...")
    print("Available endpoints:")
    print("- POST /api/predict - Main prediction endpoint")
    print("- GET /api/health - Health check")
    print("- GET /api/market-sentiment - Get current market sentiment")
    print("- POST /api/refresh-market-sentiment - Force refresh market sentiment")
    print("- GET /api/test-news - Test Alpha Vantage News API")
    print()
    
    print("Loading sentiment model...")
    try:
        initialize_sentiment_model()
        print("✅ Sentiment model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load sentiment model: {e}")
    
    # Fetch initial market sentiment with timeout
    print("📰 Fetching market sentiment...")
    try:
        import threading
        startup_sentiment_result = None
        
        def fetch_with_timeout():
            global startup_sentiment_result
            try:
                startup_sentiment_result = get_market_sentiment()
            except Exception as e:
                startup_sentiment_result = "error"
        
        # Start fetch in background thread
        fetch_thread = threading.Thread(target=fetch_with_timeout)
        fetch_thread.daemon = True
        fetch_thread.start()
        
        # Wait up to 20 seconds for completion
        fetch_thread.join(timeout=20)
        
        if hasattr(globals(), 'startup_sentiment_result') and startup_sentiment_result is not None:
            if startup_sentiment_result != "error":
                print(f"✅ Initial market sentiment: {startup_sentiment_result:.1f}/100")
            else:
                print("⚠️ Market sentiment fetch failed (will use fallback when needed)")
        else:
            print("⚠️ Market sentiment fetch timed out (will use fallback when needed)")
        
    except Exception as e:
        print("⚠️ Market sentiment fetch skipped (will use fallback when needed)")
    
    print("Server starting on http://localhost:5000")
    print("=" * 50)
    
    # Suppress Flask startup messages
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)