"""
Test script to demonstrate the new prediction bounds checking functionality.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the bounds checking function
import sys
sys.path.append('.')
from backend_server import validate_prediction_bounds

def test_bounds_checking():
    """Test the prediction bounds checking with various scenarios."""
    
    print("=== Testing Prediction Bounds Checking ===\n")
    
    # Create sample historical data for context
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic daily returns (normally distributed around 0)
    daily_returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = 100 * (1 + daily_returns).cumprod()
    
    historical_data = pd.DataFrame({
        'Close': prices
    }, index=dates)
    
    # Test cases: [prediction, description]
    test_cases = [
        (0.005, "Normal small positive prediction (0.5%)"),
        (-0.012, "Normal small negative prediction (-1.2%)"),
        (0.08, "Large but reasonable positive prediction (8%)"),
        (-0.09, "Large but reasonable negative prediction (-9%)"),
        (0.25, "Extreme positive prediction (25%) - should be clamped"),
        (-0.30, "Extreme negative prediction (-30%) - should be clamped"),
        (0.15, "Borderline large prediction (15%) - should be damped"),
        (0.0001, "Tiny prediction (0.01%) - might be increased"),
        (0.0, "Zero prediction - should remain zero"),
    ]
    
    print("Testing with strict mode (conservative bounds):")
    print("-" * 60)
    
    for prediction, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Original: {prediction:.6f} ({prediction*100:.3f}%)")
        
        # Test with historical data
        validated, warnings = validate_prediction_bounds(
            prediction, "TEST_TICKER", historical_data, strict_mode=True
        )
        
        print(f"Validated: {validated:.6f} ({validated*100:.3f}%)")
        
        if warnings:
            print(f"Warnings: {warnings[0]}")
        else:
            print("Warnings: None")
        
        # Show the change
        change = validated - prediction
        if abs(change) > 1e-6:
            print(f"Change: {change:.6f} ({change*100:.3f}%)")
    
    print("\n" + "="*60)
    print("Testing without historical data (absolute bounds only):")
    print("-" * 60)
    
    extreme_cases = [
        (0.50, "Massive positive prediction (50%)"),
        (-0.40, "Massive negative prediction (-40%)"),
        (1.0, "100% gain prediction"),
        (-0.80, "80% loss prediction"),
    ]
    
    for prediction, description in extreme_cases:
        print(f"\nTest: {description}")
        print(f"Original: {prediction:.6f} ({prediction*100:.3f}%)")
        
        # Test without historical data
        validated, warnings = validate_prediction_bounds(
            prediction, "TEST_TICKER", None, strict_mode=True
        )
        
        print(f"Validated: {validated:.6f} ({validated*100:.3f}%)")
        
        if warnings:
            print(f"Warnings: {warnings[0]}")
        
        # Show the change
        change = validated - prediction
        print(f"Change: {change:.6f} ({change*100:.3f}%)")
    
    print("\n" + "="*60)
    print("Summary of bounds checking benefits:")
    print("- Prevents extreme predictions that could lead to poor investment decisions")
    print("- Uses historical data when available for context-aware validation")
    print("- Applies different strictness levels (strict mode for final predictions)")
    print("- Provides detailed warnings for transparency")
    print("- Maintains prediction sign (direction) while controlling magnitude")

if __name__ == "__main__":
    test_bounds_checking()
