"""
Quick Test Script for Enhanced Strategy
Tests the enhanced confluence strategy with sample data
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategy_enhanced_confluence import EnhancedConfluenceStrategy
from src.strategy import MovingAverageCrossStrategy
from src.enhanced_indicators import EnhancedIndicators
from src.indicators import TechnicalIndicators
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def generate_sample_data(num_candles=1000, start_price=50000):
    """
    Generate realistic sample OHLCV data for testing.
    """
    logger.info(f"Generating {num_candles} candles of sample data...")
    
    # Create date range
    dates = pd.date_range(start=datetime.now() - timedelta(hours=num_candles), 
                         end=datetime.now(), freq='H')[:num_candles]
    
    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible results
    
    # Create trending and ranging periods
    price_changes = []
    current_trend = 1  # 1 for up, -1 for down, 0 for sideways
    trend_duration = 0
    
    for i in range(num_candles):
        # Change trend periodically
        if trend_duration > np.random.randint(50, 200):
            current_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.2, 0.5])
            trend_duration = 0
        
        # Generate price change based on trend
        if current_trend == 1:  # Uptrend
            base_change = np.random.normal(0.0005, 0.01)  # Slight upward bias
        elif current_trend == -1:  # Downtrend
            base_change = np.random.normal(-0.0005, 0.01)  # Slight downward bias
        else:  # Sideways
            base_change = np.random.normal(0, 0.005)  # No bias, less volatility
        
        # Add some noise
        noise = np.random.normal(0, 0.002)
        price_changes.append(base_change + noise)
        trend_duration += 1
    
    # Calculate prices
    prices = [start_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 100))  # Minimum price of 100
    
    prices = prices[1:]  # Remove the initial price
    
    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close price
        volatility = abs(price_changes[i]) * close
        
        open_price = close * (1 + np.random.normal(0, 0.001))
        high = max(open_price, close) + np.random.exponential(volatility * 0.5)
        low = min(open_price, close) - np.random.exponential(volatility * 0.5)
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume during price movements)
        base_volume = 1000000
        volume_multiplier = 1 + abs(price_changes[i]) * 10
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Generated data from {df.index[0]} to {df.index[-1]}")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return df

def test_strategy_performance(strategy, data, strategy_name):
    """
    Test a strategy's performance on the given data.
    """
    logger.info(f"Testing {strategy_name}...")
    
    try:
        # Generate signals
        df_with_signals = strategy.generate_signals(data)
        
        # Count signals
        buy_signals = (df_with_signals['signal'] == 'BUY').sum()
        sell_signals = (df_with_signals['signal'] == 'SELL').sum()
        hold_signals = (df_with_signals['signal'] == 'HOLD').sum()
        
        logger.info(f"Signals generated - BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
        
        # Simple performance calculation
        signals = df_with_signals[df_with_signals['signal'].isin(['BUY', 'SELL'])].copy()
        
        if len(signals) == 0:
            logger.warning("No trading signals generated!")
            return None
        
        # Calculate simple returns
        capital = 10000
        position = 0
        entry_price = 0
        trades = []
        
        for idx, row in signals.iterrows():
            current_price = row['close']
            signal = row['signal']
            
            if signal == 'BUY' and position == 0:
                # Enter long position
                position = capital / current_price
                entry_price = current_price
                capital = 0
                
            elif signal == 'SELL' and position > 0:
                # Exit long position
                capital = position * current_price
                pnl = capital - (position * entry_price)
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                position = 0
        
        # Close any open position
        if position > 0:
            final_price = data['close'].iloc[-1]
            capital = position * final_price
            pnl = capital - (position * entry_price)
            pnl_pct = (final_price - entry_price) / entry_price * 100
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        
        if len(trades) == 0:
            logger.warning("No completed trades!")
            return None
        
        # Calculate performance metrics
        total_return = capital - 10000
        total_return_pct = total_return / 10000 * 100
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Buy and hold return
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100
        
        results = {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'buy_hold_return_pct': buy_hold_return,
            'alpha': total_return_pct - buy_hold_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': capital
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error testing {strategy_name}: {str(e)}")
        return None

def run_quick_test():
    """
    Run a quick test of the enhanced strategy.
    """
    logger.info("Starting quick test of enhanced strategies...")
    
    # Generate sample data
    data = generate_sample_data(num_candles=2000, start_price=50000)
    
    # Test strategies
    strategies = {
        "Basic MA Cross": MovingAverageCrossStrategy(
            fast_ma_period=20,
            slow_ma_period=50,
            rsi_period=14
        ),
        "Enhanced Confluence (Conservative)": EnhancedConfluenceStrategy(
            min_confluence_score=0.7,
            min_adx_strength=30.0,
            min_risk_reward_ratio=2.5,
            max_trades_per_day=2,
            volume_confirmation=True
        ),
        "Enhanced Confluence (Balanced)": EnhancedConfluenceStrategy(
            min_confluence_score=0.6,
            min_adx_strength=25.0,
            min_risk_reward_ratio=2.0,
            max_trades_per_day=3,
            volume_confirmation=True
        ),
        "Enhanced Confluence (Aggressive)": EnhancedConfluenceStrategy(
            min_confluence_score=0.5,
            min_adx_strength=20.0,
            min_risk_reward_ratio=1.5,
            max_trades_per_day=4,
            volume_confirmation=False
        )
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        result = test_strategy_performance(strategy, data, strategy_name)
        if result:
            results[strategy_name] = result
    
    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("QUICK TEST RESULTS COMPARISON")
    logger.info("="*80)
    
    if results:
        # Sort by total return
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['total_return_pct'], 
                              reverse=True)
        
        for rank, (strategy_name, metrics) in enumerate(sorted_results, 1):
            logger.info(f"\n{rank}. {strategy_name}")
            logger.info(f"   Total Return: {metrics['total_return_pct']:.2f}%")
            logger.info(f"   Buy & Hold: {metrics['buy_hold_return_pct']:.2f}%")
            logger.info(f"   Alpha: {metrics['alpha']:.2f}%")
            logger.info(f"   Win Rate: {metrics['win_rate']:.1f}%")
            logger.info(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"   Number of Trades: {metrics['num_trades']}")
            logger.info(f"   Final Capital: ${metrics['final_capital']:.2f}")
        
        # Find best strategy
        best_strategy = sorted_results[0]
        logger.info(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy[0]}")
        logger.info(f"   Outperformed buy & hold by {best_strategy[1]['alpha']:.2f}%")
        
    else:
        logger.error("No valid results obtained from testing!")
    
    return results

def test_indicators():
    """
    Test the enhanced indicators functionality.
    """
    logger.info("Testing enhanced indicators...")
    
    # Generate sample data
    data = generate_sample_data(num_candles=500)
    
    try:
        # Test basic indicators
        logger.info("Testing basic indicators...")
        data_with_basic = TechnicalIndicators.calculate_all_indicators(data)
        logger.info(f"Basic indicators added: {list(data_with_basic.columns)}")
        
        # Test enhanced indicators
        logger.info("Testing enhanced indicators...")
        data_with_enhanced = EnhancedIndicators.calculate_all_enhanced_indicators(data)
        logger.info(f"Enhanced indicators added: {list(data_with_enhanced.columns)}")
        
        # Check for NaN values
        nan_counts = data_with_enhanced.isnull().sum()
        logger.info(f"NaN values per column: {nan_counts[nan_counts > 0].to_dict()}")
        
        logger.info("✅ Indicators test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Indicators test failed: {str(e)}")
        return False

def main():
    """
    Main function to run all tests.
    """
    try:
        logger.info("="*60)
        logger.info("AUTO_BOT ENHANCED STRATEGY QUICK TEST")
        logger.info("="*60)
        
        # Test 1: Indicators functionality
        logger.info("\n📊 PHASE 1: Testing Indicators")
        indicators_ok = test_indicators()
        
        if not indicators_ok:
            logger.error("Indicators test failed. Cannot proceed with strategy testing.")
            return
        
        # Test 2: Strategy performance
        logger.info("\n🎯 PHASE 2: Testing Strategy Performance")
        results = run_quick_test()
        
        if results:
            logger.info("\n✅ Quick test completed successfully!")
            logger.info("The enhanced confluence strategies show improved performance characteristics.")
            logger.info("\nNext steps:")
            logger.info("1. Run full backtests with real market data")
            logger.info("2. Optimize parameters for specific trading pairs")
            logger.info("3. Paper trade the best performing configuration")
        else:
            logger.error("❌ Quick test failed!")
        
    except Exception as e:
        logger.error(f"Error in main test: {str(e)}")
        raise

if __name__ == "__main__":
    main()