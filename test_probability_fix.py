#!/usr/bin/env python3

import sys
sys.path.append('.')
from backend_server import download_single_ticker_data, add_features_single, predict_direction_confidence
import pandas as pd

def test_multiple_stocks():
    """Test multiple stocks to ensure we always show the higher probability"""
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'NVDA']
    
    print('Testing direction probability fix with multiple stocks...')
    print('Ensuring all probabilities shown are >= 50% (the more likely outcome)\n')
    
    for ticker in tickers:
        print(f'=== Testing {ticker} ===')
        
        # Download and process data
        raw_data = download_single_ticker_data(ticker, '2024-01-01')
        if raw_data is None or raw_data.empty:
            print(f'❌ Failed to download data for {ticker}')
            continue
            
        features_df = add_features_single(ticker, raw_data)
        if features_df is None or features_df.empty:
            print(f'❌ Failed to extract features for {ticker}')
            continue
        
        # Get direction prediction
        result = predict_direction_confidence(ticker, features_df, prediction_window=5)
        
        direction = result['direction']
        probability = result['direction_probability']
        
        print(f'✅ {ticker}: {direction.upper()} - {probability:.1f}%')
        
        # Verify the probability is >= 50%
        if probability >= 50.0:
            print(f'   ✅ Correct: Showing higher probability ({probability:.1f}% >= 50%)')
        else:
            print(f'   ❌ ERROR: Showing lower probability ({probability:.1f}% < 50%)')
            return False
            
        print()
    
    return True

if __name__ == '__main__':
    success = test_multiple_stocks()
    if success:
        print('🎉 All stocks show the higher probability correctly!')
    else:
        print('❌ Some stocks show incorrect probabilities!')
