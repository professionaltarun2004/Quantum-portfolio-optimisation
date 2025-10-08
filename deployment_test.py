#!/usr/bin/env python3
"""
Deployment readiness test script.
Run this before deploying to ensure everything works correctly.
"""

import sys
import importlib
import subprocess
import os

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing Imports...")
    
    critical_modules = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'yfinance',
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'requests'
    ]
    
    failed_imports = []
    
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_backend_modules():
    """Test backend module imports."""
    print("\n🔧 Testing Backend Modules...")
    
    backend_modules = [
        'backend.classical_optimizer',
        'backend.quantum_optimizer', 
        'backend.qubo_encoder',
        'backend.comparator',
        'backend.explanation_layer'
    ]
    
    failed_modules = []
    
    for module in backend_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_modules.append(module)
    
    return len(failed_modules) == 0

def test_app_import():
    """Test main app import without context errors."""
    print("\n📱 Testing App Import...")
    
    try:
        import app
        print("✅ App imports without context errors")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_optimization_methods():
    """Test optimization methods work."""
    print("\n⚙️ Testing Optimization Methods...")
    
    try:
        from backend.classical_optimizer import ClassicalOptimizer
        import pandas as pd
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (100, 3)),
            index=dates,
            columns=tickers
        )
        
        optimizer = ClassicalOptimizer()
        result = optimizer.optimize(returns_data, risk_tolerance=1.0, method='mean_variance')
        
        # Validate result
        if 'weights' in result and 'expected_return' in result and 'volatility' in result:
            print("✅ Optimization methods working")
            return True
        else:
            print("❌ Optimization result missing required keys")
            return False
            
    except Exception as e:
        print(f"❌ Optimization test failed: {e}")
        return False

def check_files():
    """Check required files exist."""
    print("\n📁 Checking Required Files...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'Procfile',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def main():
    """Run all deployment tests."""
    print("🚀 Deployment Readiness Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Backend Modules", test_backend_modules), 
        ("App Import", test_app_import),
        ("Optimization Test", test_optimization_methods),
        ("File Check", check_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Ready for deployment!")
        print("\n📋 Deployment Instructions:")
        print("1. Push code to GitHub")
        print("2. Go to share.streamlit.io")
        print("3. Connect your repo")
        print("4. Set main file: app.py")
        print("5. Deploy!")
        return True
    else:
        print("❌ Fix the failing tests before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)