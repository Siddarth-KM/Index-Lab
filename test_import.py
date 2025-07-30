#!/usr/bin/env python3
"""Simple test to check if backend_server imports correctly"""

try:
    import backend_server
    print("✅ Backend server imports successfully!")
    
    # Test basic function availability
    functions_to_test = [
        'ensure_iterable',
        'sanitize_for_json', 
        'calculate_market_condition',
        'train_model_for_stock',
        'add_features_to_stock'
    ]
    
    for func_name in functions_to_test:
        if hasattr(backend_server, func_name):
            print(f"✅ Function {func_name} found")
        else:
            print(f"❌ Function {func_name} missing")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
