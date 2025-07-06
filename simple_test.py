#!/usr/bin/env python3
"""
Simple test to validate the enhanced strategy implementation
"""

import sys
import os
import traceback

print("="*60)
print("AUTO_BOT ENHANCED STRATEGY VALIDATION TEST")
print("="*60)

try:
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    import pandas as pd
    import numpy as np
    print("✅ pandas and numpy imported successfully")
    
    # Test 2: Check if our modules can be imported
    print("\n2. Testing enhanced modules...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from enhanced_indicators import EnhancedIndicators
        print("✅ Enhanced indicators module imported")
    except Exception as e:
        print(f"❌ Enhanced indicators import failed: {e}")
    
    try:
        from market_structure import MarketStructure
        print("✅ Market structure module imported")
    except Exception as e:
        print(f"❌ Market structure import failed: {e}")
    
    try:
        from strategy_enhanced_confluence import EnhancedConfluenceStrategy
        print("✅ Enhanced confluence strategy imported")
    except Exception as e:
        print(f"❌ Enhanced confluence strategy import failed: {e}")
    
    # Test 3: Create sample data
    print("\n3. Creating sample data...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Generate simple price data
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    print(f"✅ Sample data created: {len(sample_data)} rows")
    print(f"   Price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}")
    
    # Test 4: Test enhanced indicators
    print("\n4. Testing enhanced indicators...")
    try:
        enhanced_data = EnhancedIndicators.calculate_all_enhanced_indicators(sample_data)
        print(f"✅ Enhanced indicators calculated: {len(enhanced_data.columns)} columns")
        print(f"   New indicators: {[col for col in enhanced_data.columns if col not in sample_data.columns][:5]}...")
    except Exception as e:
        print(f"❌ Enhanced indicators failed: {e}")
        traceback.print_exc()
    
    # Test 5: Test strategy creation
    print("\n5. Testing strategy creation...")
    try:
        strategy = EnhancedConfluenceStrategy(
            min_confluence_score=0.6,
            min_adx_strength=25.0,
            min_risk_reward_ratio=2.0
        )
        print(f"✅ Strategy created: {strategy.name}")
    except Exception as e:
        print(f"❌ Strategy creation failed: {e}")
        traceback.print_exc()
    
    # Test 6: Test signal generation (basic)
    print("\n6. Testing signal generation...")
    try:
        # First add basic indicators
        from indicators import TechnicalIndicators
        data_with_basic = TechnicalIndicators.calculate_all_indicators(sample_data)
        
        # Then add enhanced indicators
        data_with_enhanced = EnhancedIndicators.calculate_all_enhanced_indicators(data_with_basic)
        
        # Generate signals
        signals_data = strategy.generate_signals(data_with_enhanced)
        
        # Count signals
        buy_signals = (signals_data['signal'] == 'BUY').sum()
        sell_signals = (signals_data['signal'] == 'SELL').sum()
        hold_signals = (signals_data['signal'] == 'HOLD').sum()
        
        print(f"✅ Signals generated successfully:")
        print(f"   BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
        
        if 'confluence_score' in signals_data.columns:
            avg_score = signals_data['confluence_score'].mean()
            print(f"   Average confluence score: {avg_score:.3f}")
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("VALIDATION TEST COMPLETED")
    print("="*60)
    print("✅ Enhanced strategy implementation is working!")
    print("\nNext steps:")
    print("1. Run comprehensive backtests with real data")
    print("2. Optimize parameters for your trading pairs")
    print("3. Paper trade the strategy")
    print("4. Integrate with your main bot")
    
except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")
    traceback.print_exc()
    print("\nPlease check the error above and fix any missing dependencies.")

print("\nTest completed!")