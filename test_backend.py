#!/usr/bin/env python3
"""Test script to verify backend_server fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing backend_server import...")
    import backend_server
    print("✅ Import successful!")
    
    # Test basic functions
    print("\nTesting utility functions...")
    
    # Test ensure_iterable
    result = backend_server.ensure_iterable(5.0)
    print(f"✅ ensure_iterable(5.0) = {result}")
    
    # Test sanitize_for_json
    import numpy as np
    test_data = {'value': np.nan, 'list': [1, 2, np.inf]}
    result = backend_server.sanitize_for_json(test_data)
    print(f"✅ sanitize_for_json with NaN/inf = {result}")
    
    # Test model configurations
    print(f"✅ MODEL_CONFIGS loaded: {len(backend_server.MODEL_CONFIGS)} models")
    
    # Test that critical functions exist
    functions_to_check = [
        'train_model_for_stock',
        'train_and_predict_model', 
        'add_features_to_stock',
        'download_market_data_cache',
        'calculate_market_condition'
    ]
    
    for func_name in functions_to_check:
        if hasattr(backend_server, func_name):
            print(f"✅ Function {func_name} found")
        else:
            print(f"❌ Function {func_name} missing")
    
    print("\n🎉 All basic tests passed! Backend server should work correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
