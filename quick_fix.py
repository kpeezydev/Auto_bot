#!/usr/bin/env python3
"""
Quick fix script to install dependencies and test the enhanced strategy
"""

import subprocess
import sys
import os

print("="*60)
print("AUTO_BOT QUICK FIX - INSTALLING DEPENDENCIES")
print("="*60)

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"\nInstalling {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def test_import(package, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        print(f"✅ {package} is available")
        return True
    except ImportError:
        print(f"❌ {package} is not available")
        return False

# Required packages
packages = [
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn")
]

print("\n1. Checking current package status...")
missing_packages = []

for package, import_name in packages:
    if not test_import(package, import_name):
        missing_packages.append(package)

if not missing_packages:
    print("\n✅ All required packages are already installed!")
else:
    print(f"\n📦 Installing missing packages: {missing_packages}")
    
    for package in missing_packages:
        install_package(package)

print("\n2. Testing enhanced strategy import...")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.strategy_enhanced_confluence import EnhancedConfluenceStrategy
    print("✅ Enhanced confluence strategy imported successfully")
    
    # Test strategy creation
    strategy = EnhancedConfluenceStrategy(
        min_confluence_score=0.6,
        min_adx_strength=25.0,
        min_risk_reward_ratio=2.0,
        max_trades_per_day=3
    )
    print(f"✅ Strategy created: {strategy.name}")
    
except Exception as e:
    print(f"❌ Strategy import failed: {e}")
    print("\nUsing simplified version without scipy...")
    
    try:
        # Test with simplified market structure
        from src.market_structure_simple import MarketStructure
        print("✅ Simplified market structure available")
        
        from src.strategy_enhanced_confluence import EnhancedConfluenceStrategy
        strategy = EnhancedConfluenceStrategy(
            min_confluence_score=0.6,
            min_adx_strength=25.0,
            min_risk_reward_ratio=2.0,
            max_trades_per_day=3
        )
        print(f"✅ Strategy created with simplified version: {strategy.name}")
        
    except Exception as e2:
        print(f"❌ Even simplified version failed: {e2}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("QUICK FIX COMPLETED")
print("="*60)

print("\n🚀 Try running your enhanced strategy now:")
print("python main.py --strategy enhanced_confluence --mode paper --pair BTC/USDT --run-once")

print("\nIf you still get errors, the simplified version should work without scipy.")
print("The enhanced strategy will use basic swing point detection instead of scipy.")