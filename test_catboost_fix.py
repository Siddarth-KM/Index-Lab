#!/usr/bin/env python3

import sys
sys.path.append('.')
from backend_server import download_single_ticker_data, add_features_single, predict_direction_confidence
import pandas as pd

def test_catboost_fix():
    """Test the comprehensive CatBoost fix"""
    print('Testing AAPL with comprehensive CatBoost fix...')
    
    # Download raw data
    raw_data = download_single_ticker_data('AAPL', '2024-01-01')
    if raw_data is None or raw_data.empty:
        print('❌ Failed to download raw data')
        return False
        
    print(f'✅ Downloaded raw data: {raw_data.shape}')
    
    # Extract features
    features_df = add_features_single('AAPL', raw_data)
    if features_df is None or features_df.empty:
        print('❌ Failed to extract features')
        return False
        
    print(f'✅ Extracted features: {features_df.shape}')
    
    # Test direction prediction
    result = predict_direction_confidence('AAPL', features_df, prediction_window=5)
    if result is None:
        print('❌ Failed to predict direction')
        return False
        
    print('✅ Direction prediction result:')
    print(f'  - Direction: {result["direction"]}')
    print(f'  - Probability: {result["probability"]:.1f}%')
    print(f'  - Has error: {"error" in result and result.get("error") is not None}')
    
    # Check if error indicates CatBoost failure
    if 'error' in result and result.get('error'):
        if 'floating point numerical type' in str(result['error']):
            print('❌ CatBoost categorical features error still occurring!')
            return False
        else:
            print(f'⚠️  Other error: {result["error"]}')
    else:
        print('✅ No CatBoost categorical features error!')
        
    return True

if __name__ == '__main__':
    success = test_catboost_fix()
    if success:
        print('\n🎉 CatBoost fix test completed successfully!')
    else:
        print('\n❌ CatBoost fix test failed!')
