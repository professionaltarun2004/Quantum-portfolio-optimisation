#!/usr/bin/env python3
"""
Launch script for the Quantum vs Classical Portfolio Optimization app.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'qiskit', 'cvxpy', 'numpy', 'pandas', 
        'plotly', 'yfinance', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main function to launch the app."""
    print("🚀 Quantum vs Classical Portfolio Optimization")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found in current directory")
        sys.exit(1)
    
    print("\n🌐 Starting Streamlit application...")
    print("📍 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚡ To stop the app, press Ctrl+C in this terminal")
    print("=" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()