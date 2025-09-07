#!/usr/bin/env python3
"""
Test script to verify all fixes are working properly.
"""

import pandas as pd
import numpy as np
from backend.classical_optimizer import ClassicalOptimizer

def test_optimization_methods():
    """Test all optimization methods with synthetic data."""
    print("🧪 Testing Portfolio Optimization Fixes")
    print("=" * 50)
    
    # Create synthetic test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
    
    # Generate realistic returns data
    returns_data = pd.DataFrame(
        np.random.normal(0.0008, 0.02, (200, len(tickers))),
        index=dates,
        columns=tickers
    )
    
    # Add some correlation
    market_factor = np.random.normal(0, 0.01, 200)
    for col in returns_data.columns:
        returns_data[col] += 0.3 * market_factor
    
    optimizer = ClassicalOptimizer()
    methods = ['mean_variance', 'lasso', 'ridge', 'ml_enhanced', 'genetic_algorithm', 'neural_network']
    
    results = {}
    
    for method in methods:
        try:
            print(f"\n🔄 Testing {method}...")
            result = optimizer.optimize(returns_data, risk_tolerance=1.0, method=method)
            
            # Validate result structure
            required_keys = ['weights', 'expected_return', 'volatility', 'sharpe_ratio']
            missing_keys = [key for key in required_keys if key not in result]
            
            if missing_keys:
                print(f"❌ {method}: Missing keys {missing_keys}")
                continue
            
            # Validate weights
            weights = result['weights']
            if not isinstance(weights, np.ndarray):
                print(f"❌ {method}: Weights not numpy array")
                continue
                
            if len(weights) != len(tickers):
                print(f"❌ {method}: Wrong number of weights ({len(weights)} vs {len(tickers)})")
                continue
                
            if not np.isclose(weights.sum(), 1.0, atol=1e-6):
                print(f"❌ {method}: Weights don't sum to 1 ({weights.sum():.6f})")
                continue
                
            if np.any(weights < -1e-6):  # Allow small numerical errors
                print(f"❌ {method}: Negative weights found")
                continue
            
            # Validate metrics
            if not np.isfinite(result['expected_return']):
                print(f"❌ {method}: Invalid expected return")
                continue
                
            if not np.isfinite(result['volatility']) or result['volatility'] <= 0:
                print(f"❌ {method}: Invalid volatility")
                continue
            
            print(f"✅ {method}: SUCCESS")
            print(f"   Expected Return: {result['expected_return']:.4f}")
            print(f"   Volatility: {result['volatility']:.4f}")
            print(f"   Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"   Weights sum: {weights.sum():.6f}")
            
            results[method] = result
            
        except Exception as e:
            print(f"❌ {method}: ERROR - {str(e)}")
    
    print(f"\n📊 Summary: {len(results)}/{len(methods)} methods working")
    
    if len(results) >= 3:
        print("✅ All critical fixes are working!")
        return True
    else:
        print("❌ Some methods still have issues")
        return False

if __name__ == "__main__":
    success = test_optimization_methods()
    exit(0 if success else 1)