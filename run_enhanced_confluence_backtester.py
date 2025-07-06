"""
Enhanced Confluence Strategy Backtester
Test the new multi-indicator confluence strategy for improved winning rate
"""

import logging
import argparse
import sys
from datetime import datetime, timedelta

from src.strategy_enhanced_confluence import EnhancedConfluenceStrategy
from src.backtester import Backtester
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtester with enhanced confluence strategy")
    
    parser.add_argument("--exchange", type=str, default="binance",
                        help="Exchange to use (default: binance)")
    parser.add_argument("--symbol", type=str, default="BTC/USDT",
                        help="Trading pair symbol (default: BTC/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe for analysis (default: 1h)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Number of candles to fetch (default: 5000)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital for backtest (default: 10000.0)")
    parser.add_argument("--commission", type=float, default=0.1,
                        help="Commission percentage (default: 0.1)")
    parser.add_argument("--risk", type=float, default=1.0,
                        help="Risk percentage per trade (default: 1.0)")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    parser.add_argument("--start-date", type=str,
                        default=start_date.strftime('%Y-%m-%d'),
                        help="Start date for backtest (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                        default=end_date.strftime('%Y-%m-%d'),
                        help="End date for backtest (format: YYYY-MM-DD)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of results")
    
    # Enhanced Confluence Strategy Parameters
    parser.add_argument("--confluence-threshold", type=float, default=0.6,
                        help="Minimum confluence score for signal (default: 0.6)")
    parser.add_argument("--trend-strength-threshold", type=float, default=0.3,
                        help="Minimum trend strength (default: 0.3)")
    parser.add_argument("--min-adx", type=float, default=25.0,
                        help="Minimum ADX for trend strength (default: 25.0)")
    parser.add_argument("--risk-reward-ratio", type=float, default=2.0,
                        help="Minimum risk-reward ratio (default: 2.0)")
    parser.add_argument("--atr-multiplier", type=float, default=2.0,
                        help="ATR multiplier for stop loss (default: 2.0)")
    parser.add_argument("--max-trades-per-day", type=int, default=3,
                        help="Maximum trades per day (default: 3)")
    parser.add_argument("--min-trade-spacing", type=int, default=4,
                        help="Minimum hours between trades (default: 4)")
    parser.add_argument("--disable-volume-confirmation", action="store_true",
                        help="Disable volume confirmation")
    parser.add_argument("--disable-dynamic-stops", action="store_true",
                        help="Disable dynamic stop losses")
    
    return parser.parse_args()

def run_comparison_test(args):
    """Run comparison between different strategies."""
    logger.info("Running strategy comparison test...")
    
    strategies = {
        "Enhanced Confluence (Conservative)": EnhancedConfluenceStrategy(
            min_confluence_score=0.7,
            trend_strength_threshold=0.4,
            min_adx_strength=30.0,
            min_risk_reward_ratio=2.5,
            max_trades_per_day=2
        ),
        "Enhanced Confluence (Balanced)": EnhancedConfluenceStrategy(
            min_confluence_score=0.6,
            trend_strength_threshold=0.3,
            min_adx_strength=25.0,
            min_risk_reward_ratio=2.0,
            max_trades_per_day=3
        ),
        "Enhanced Confluence (Aggressive)": EnhancedConfluenceStrategy(
            min_confluence_score=0.5,
            trend_strength_threshold=0.2,
            min_adx_strength=20.0,
            min_risk_reward_ratio=1.5,
            max_trades_per_day=4
        )
    }
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        logger.info(f"\nTesting {strategy_name}...")
        
        backtester = Backtester(
            strategy=strategy,
            initial_capital=args.capital,
            commission_pct=args.commission,
            risk_per_trade_pct=args.risk
        )
        
        # Fetch data
        data = backtester.fetch_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )
        
        if len(data) == 0:
            logger.error(f"No data available for {strategy_name}")
            continue
        
        # Run backtest
        result = backtester.run_backtest(data)
        results[strategy_name] = result
        
        # Log results
        performance = result['performance']
        logger.info(f"{strategy_name} Results:")
        logger.info(f"  Total Return: {performance['return_pct']:.2f}%")
        logger.info(f"  Win Rate: {performance['win_rate']:.2f}%")
        logger.info(f"  Profit Factor: {performance['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        logger.info(f"  Number of Trades: {len(result['trades'])}")
    
    # Print comparison summary
    logger.info("\n" + "="*60)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("="*60)
    
    for strategy_name, result in results.items():
        performance = result['performance']
        logger.info(f"\n{strategy_name}:")
        logger.info(f"  Return: {performance['return_pct']:.2f}% | Win Rate: {performance['win_rate']:.2f}% | "
                   f"Profit Factor: {performance['profit_factor']:.2f} | Max DD: {performance['max_drawdown_pct']:.2f}%")

def main():
    """Main function to run the enhanced confluence backtester."""
    args = parse_arguments()
    
    try:
        logger.info("Starting Enhanced Confluence Strategy Backtester")
        logger.info(f"Testing on {args.symbol} from {args.start_date} to {args.end_date}")
        
        # Check if comparison mode
        if len(sys.argv) > 1 and "--compare" in sys.argv:
            run_comparison_test(args)
            return
        
        # Create enhanced confluence strategy
        strategy = EnhancedConfluenceStrategy(
            min_confluence_score=args.confluence_threshold,
            trend_strength_threshold=args.trend_strength_threshold,
            volume_confirmation=not args.disable_volume_confirmation,
            risk_per_trade_pct=args.risk,
            min_risk_reward_ratio=args.risk_reward_ratio,
            use_dynamic_stops=not args.disable_dynamic_stops,
            atr_stop_multiplier=args.atr_multiplier,
            min_adx_strength=args.min_adx,
            max_trades_per_day=args.max_trades_per_day,
            min_trade_spacing_hours=args.min_trade_spacing
        )
        
        logger.info(f"Strategy Configuration:")
        logger.info(f"  Confluence Threshold: {args.confluence_threshold}")
        logger.info(f"  Trend Strength Threshold: {args.trend_strength_threshold}")
        logger.info(f"  Min ADX: {args.min_adx}")
        logger.info(f"  Risk-Reward Ratio: {args.risk_reward_ratio}")
        logger.info(f"  Volume Confirmation: {not args.disable_volume_confirmation}")
        logger.info(f"  Dynamic Stops: {not args.disable_dynamic_stops}")
        logger.info(f"  Max Trades/Day: {args.max_trades_per_day}")
        
        # Create backtester
        backtester = Backtester(
            strategy=strategy,
            initial_capital=args.capital,
            commission_pct=args.commission,
            risk_per_trade_pct=args.risk
        )
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        data = backtester.fetch_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )
        
        if len(data) == 0:
            logger.error("No data available for backtesting. Check your parameters.")
            return
        
        logger.info(f"Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtester.run_backtest(data)
        
        # Plot results if enabled
        if not args.no_plot:
            try:
                backtester.plot_results(results)
            except Exception as e:
                logger.warning(f"Could not plot results: {str(e)}")
        
        # Print detailed performance summary
        performance = results['performance']
        trades = results['trades']
        
        logger.info("\n" + "="*60)
        logger.info("ENHANCED CONFLUENCE STRATEGY BACKTEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"\nPeriod: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Symbol: {args.symbol} | Timeframe: {args.timeframe}")
        logger.info(f"Initial Capital: ${args.capital:,.2f}")
        logger.info(f"Final Capital: ${backtester.capital:,.2f}")
        
        logger.info(f"\nPERFORMANCE METRICS:")
        logger.info(f"Total Return: ${performance['total_return']:,.2f} ({performance['return_pct']:.2f}%)")
        logger.info(f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%")
        logger.info(f"Alpha (vs Buy & Hold): {performance['return_pct'] - performance['buy_hold_return']:.2f}%")
        
        logger.info(f"\nTRADE STATISTICS:")
        logger.info(f"Total Trades: {len(trades)}")
        logger.info(f"Win Rate: {performance['win_rate']:.2f}%")
        logger.info(f"Average Profit: ${performance['avg_profit']:,.2f}")
        logger.info(f"Average Loss: ${performance['avg_loss']:,.2f}")
        logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
        logger.info(f"Largest Win: ${performance.get('largest_win', 0):,.2f}")
        logger.info(f"Largest Loss: ${performance.get('largest_loss', 0):,.2f}")
        
        logger.info(f"\nRISK METRICS:")
        logger.info(f"Maximum Drawdown: {performance['max_drawdown_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}")
        
        # Trade frequency analysis
        if len(trades) > 0:
            trade_duration_hours = []
            for trade in trades:
                if hasattr(trade, 'exit_time') and hasattr(trade, 'entry_time'):
                    duration = trade.exit_time - trade.entry_time
                    hours = duration.total_seconds() / 3600
                    trade_duration_hours.append(hours)
            
            if trade_duration_hours:
                avg_duration = sum(trade_duration_hours) / len(trade_duration_hours)
                logger.info(f"Average Trade Duration: {avg_duration:.1f} hours")
        
        logger.info(f"\nSTRATEGY EFFECTIVENESS:")
        if performance['return_pct'] > performance['buy_hold_return']:
            logger.info("✅ Strategy OUTPERFORMED buy & hold")
        else:
            logger.info("❌ Strategy UNDERPERFORMED buy & hold")
        
        if performance['win_rate'] > 50:
            logger.info(f"✅ Win rate above 50%: {performance['win_rate']:.1f}%")
        else:
            logger.info(f"❌ Win rate below 50%: {performance['win_rate']:.1f}%")
        
        if performance['profit_factor'] > 1.5:
            logger.info(f"✅ Good profit factor: {performance['profit_factor']:.2f}")
        else:
            logger.info(f"⚠️  Low profit factor: {performance['profit_factor']:.2f}")
        
        logger.info("\nBacktest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running backtester: {str(e)}")
        raise

if __name__ == "__main__":
    main()