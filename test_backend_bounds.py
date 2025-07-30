"""
Test the updated backend with bounds checking by making a prediction request.
"""

import sys
sys.path.append('.')
import yfinance as yf
from backend_server import train_model_for_stock, download_market_data_cache, add_features_to_stock
from datetime import datetime, timedelta
import pandas as pd

def test_backend_with_bounds():
    """Test the backend prediction pipeline with bounds checking."""
    
    print("=== Testing Backend with Prediction Bounds Checking ===\n")
    
    # Download some test data
    ticker = "AAPL"
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Downloading data for {ticker} from {start_date}...")
    
    try:
        # Download stock data
        df = yf.download(ticker, start=start_date, progress=False)
        if df is None or len(df) < 50:
            print("Failed to download sufficient data")
            return
        
        print(f"Downloaded {len(df)} days of data")
        
        # Download market data for cross-asset features
        print("Downloading market reference data...")
        market_data = download_market_data_cache(start_date, force_refresh=False)
        
        # Add features
        print("Adding technical features...")
        df_with_features = add_features_to_stock(ticker, df, prediction_window=5, market_data_cache=market_data)
        
        if df_with_features is None:
            print("Failed to add features")
            return
        
        print(f"Added features, final dataset: {len(df_with_features)} rows, {len(df_with_features.columns)} columns")
        
        # Test prediction with multiple models
        model_ids = [1, 2, 6]  # XGBoost, Random Forest, Bayesian Ridge (conservative models)
        
        print(f"\nTesting prediction with models: {model_ids}")
        print("This should demonstrate bounds checking in action...")
        
        prediction_result = train_model_for_stock(
            ticker, 
            df_with_features, 
            model_ids, 
            regime='sideways',  # Conservative regime
            regime_strength=0.5
        )
        
        if prediction_result:
            print(f"\n=== Prediction Results for {ticker} ===")
            print(f"Final Prediction: {prediction_result.get('prediction', 0):.6f} ({prediction_result.get('percentage', 0):.3f}%)")
            print(f"Confidence: {prediction_result.get('confidence', 0):.3f}")
            print(f"Valid Models: {prediction_result.get('valid_models', 0)}/{prediction_result.get('total_models', 0)}")
            
            # Show if bounds checking was applied
            if prediction_result.get('validation_applied', False):
                print(f"✓ Bounds validation was applied")
                if prediction_result.get('validation_warnings'):
                    print(f"Warnings: {len(prediction_result.get('validation_warnings', []))}")
            else:
                print("✓ Prediction within normal bounds, no validation needed")
            
            # Show individual model predictions
            if 'individual_predictions' in prediction_result:
                print("\nIndividual Model Predictions:")
                for model_name, pred in prediction_result['individual_predictions'].items():
                    print(f"  {model_name}: {pred:.6f} ({pred*100:.3f}%)")
            
        else:
            print("Prediction failed")
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backend_with_bounds()
