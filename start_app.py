#!/usr/bin/env python3
"""
Simple script to start the Streamlit app locally.
Use this instead of complex command line arguments.
"""

import subprocess
import sys
import os

def start_app():
    """Start the Streamlit app with proper configuration."""
    print("🚀 Starting Quantum vs Classical Portfolio Optimization App")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Start the app
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8502",
        "--server.address", "localhost"
    ]
    
    print(f"🌐 Starting app on http://localhost:8502")
    print("📍 The app will open in your default browser")
    print("⚡ To stop the app, press Ctrl+C")
    print("=" * 60)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_app()