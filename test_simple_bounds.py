"""
Simple test to demonstrate prediction bounds checking without complex feature engineering.
"""

import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from backend_server import validate_prediction_bounds, train_model_for_stock

def create_simple_test_data():
    """Create simple test data with basic features."""
    
    # Create 100 days of synthetic data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
    prices = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'Close': prices,
        'High': prices * (1 + np.abs(np.random.normal(0, 0.005, 100))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, 100))),
        'Volume': np.random.randint(1000000, 5000000, 100),
    }, index=dates)
    
    # Add some simple features
    df['returns'] = df['Close'].pct_change()
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['volatility'] = df['returns'].rolling(10).std()
    df['momentum'] = df['Close'] / df['Close'].shift(5) - 1
    df['rsi_proxy'] = (df['Close'] - df['Close'].rolling(14).min()) / (df['Close'].rolling(14).max() - df['Close'].rolling(14).min()) * 100
    
    return df.dropna()

def test_simple_prediction_with_bounds():
    """Test prediction bounds checking with simple synthetic data."""
    
    print("=== Simple Prediction Bounds Testing ===\n")
    
    # Create test data
    df = create_simple_test_data()
    print(f"Created test dataset with {len(df)} rows and {len(df.columns)} columns")
    
    # Manually create some extreme predictions to test bounds checking
    test_predictions = {
        'Model_1': {'avg_pred': 0.25},     # 25% - should be clamped
        'Model_2': {'avg_pred': -0.18},    # -18% - should be damped  
        'Model_3': {'avg_pred': 0.08},     # 8% - might be reduced based on historical data
        'Model_4': {'avg_pred': 0.003},    # 0.3% - normal, should pass
        'Model_5': {'avg_pred': -0.35},    # -35% - should be clamped
    }
    
    print("Testing individual model prediction bounds:")
    print("-" * 50)
    
    for model_name, pred_data in test_predictions.items():
        original_pred = pred_data['avg_pred']
        validated_pred, warnings = validate_prediction_bounds(
            original_pred, f"TEST_{model_name}", df, strict_mode=False
        )
        
        print(f"{model_name}: {original_pred:.6f} → {validated_pred:.6f} ({validated_pred*100:.3f}%)")
        if warnings:
            print(f"  Warning: {warnings[0]}")
        print()
    
    # Now test the ensemble prediction bounds checking
    print("\nTesting ensemble prediction bounds:")
    print("-" * 40)
    
    # Simulate what happens in train_model_for_stock
    valid_predictions = [pred['avg_pred'] for pred in test_predictions.values() 
                        if abs(pred['avg_pred']) > 1e-6]
    
    if valid_predictions:
        # Calculate ensemble prediction
        ensemble_pred = np.median(valid_predictions)
        print(f"Raw ensemble (median): {ensemble_pred:.6f} ({ensemble_pred*100:.3f}%)")
        
        # Apply confidence damping
        pred_spread = np.max(valid_predictions) - np.min(valid_predictions)
        confidence = max(0.1, 1.0 - (pred_spread * 10))
        damped_pred = ensemble_pred * confidence
        print(f"After confidence damping: {damped_pred:.6f} ({damped_pred*100:.3f}%)")
        print(f"Confidence: {confidence:.3f}, Spread: {pred_spread:.6f}")
        
        # Apply final bounds validation (strict mode)
        final_pred, final_warnings = validate_prediction_bounds(
            damped_pred, "ENSEMBLE", df, strict_mode=True
        )
        
        print(f"Final validated prediction: {final_pred:.6f} ({final_pred*100:.3f}%)")
        if final_warnings:
            print(f"Final warnings: {final_warnings}")
        
        print(f"\nTotal change from raw ensemble: {final_pred - ensemble_pred:.6f} ({(final_pred - ensemble_pred)*100:.3f}%)")
    
    print("\n" + "="*60)
    print("Benefits of prediction bounds checking:")
    print("✓ Prevents extreme predictions that could cause financial losses")
    print("✓ Uses historical context when available for intelligent capping")  
    print("✓ Applies progressive damping (soft limits before hard limits)")
    print("✓ Maintains prediction direction while controlling magnitude")
    print("✓ Provides transparency with detailed warnings")
    print("✓ Different strictness levels for individual vs ensemble predictions")

if __name__ == "__main__":
    test_simple_prediction_with_bounds()
