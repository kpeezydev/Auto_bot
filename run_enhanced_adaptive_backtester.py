import logging
import argparse
from datetime import datetime, timedelta

from src.strategy_enhanced_adaptive import EnhancedAdaptiveStrategy
from src.backtester import Backtester
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtester with enhanced adaptive trading strategy")
    
    parser.add_argument("--exchange", type=str, default="binance",
                        help="Exchange to use (default: binance)")
    parser.add_argument("--symbol", type=str, default="SOL/USDT",
                        help="Trading pair symbol (default: SOL/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe for analysis (default: 1h)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Number of candles to fetch (default: 10000)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital for backtest (default: 10000.0)")
    parser.add_argument("--commission", type=float, default=0.1,
                        help="Commission percentage (default: 0.1)")
    parser.add_argument("--risk", type=float, default=1.0,
                        help="Risk percentage per trade (default: 1.0)")
    # Ensure we always use last 730 days (2 years) by default
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    parser.add_argument("--start-date", type=str,
                        default=start_date.strftime('%Y-%m-%d'),
                        help="Start date for backtest (format: YYYY-MM-DD, default: 365 days ago)")
    parser.add_argument("--end-date", type=str,
                        default=end_date.strftime('%Y-%m-%d'),
                        help="End date for backtest (format: YYYY-MM-DD, default: today)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of results")
    
    # Add parameters specific to the enhanced adaptive strategy
    parser.add_argument("--fast-ma", type=int, default=12,       # Changed default
                        help="Base fast MA period (default: 12)")
    parser.add_argument("--slow-ma", type=int, default=26,       # Changed default
                        help="Base slow MA period (default: 26)")
    parser.add_argument("--rsi-period", type=int, default=14,
                        help="Base RSI period (default: 14)")
    parser.add_argument("--rsi-overbought", type=int, default=70, # Reverted default
                        help="Base RSI overbought threshold (default: 70)")
    parser.add_argument("--rsi-oversold", type=int, default=30, # Reverted default
                        help="Base RSI oversold threshold (default: 30)")
    parser.add_argument("--trend-ema", type=int, default=50,
                        help="Trend EMA period (default: 50)")
    parser.add_argument("--volatility-lookback", type=int, default=50,
                        help="Lookback period for volatility calculation (default: 50)")
    parser.add_argument("--regime-lookback", type=int, default=30,
                        help="Lookback period for market regime detection (default: 30)")
    parser.add_argument("--no-adaptive", action="store_true", default=True, # Changed default to disable adaptive
                        help="Disable adaptive parameter adjustment")
    parser.add_argument("--trailing-stop", type=float, default=2.5,    # Reverted default
                        help="Trailing stop percentage (default: 2.5)")
    parser.add_argument("--no-trailing-stop", action="store_true",
                        help="Disable trailing stop loss")
    parser.add_argument("--min-trade-duration", type=int, default=4,
                        help="Minimum number of candles to hold a trade (default: 4)")
    parser.add_argument("--max-trades-per-day", type=int, default=2,
                        help="Maximum number of trades per day (default: 2)")
    
    return parser.parse_args()

def main():
    """Main function to run the backtester with enhanced adaptive strategy."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        logger.info("Starting backtester with enhanced adaptive strategy")
        
        # Create enhanced adaptive strategy instance with parameters from command line
        strategy = EnhancedAdaptiveStrategy(
            fast_ma_period=args.fast_ma,
            slow_ma_period=args.slow_ma,
            rsi_period=args.rsi_period,
            rsi_overbought=args.rsi_overbought,
            rsi_oversold=args.rsi_oversold,
            trend_ema_period=args.trend_ema,
            volatility_lookback=args.volatility_lookback,
            regime_lookback=args.regime_lookback,
            use_adaptive_parameters=not args.no_adaptive,
            use_trailing_stop=not args.no_trailing_stop,
            trailing_stop_pct=args.trailing_stop,
            min_trade_duration=args.min_trade_duration,
            max_trades_per_day=args.max_trades_per_day,
            btc_specific_optimization=False # Explicitly disable BTC optimization
        )
        
        logger.info(f"Using {strategy.name} with base parameters: "
                   f"Fast MA={args.fast_ma}, Slow MA={args.slow_ma}, "
                   f"RSI Period={args.rsi_period}, RSI Overbought={args.rsi_overbought}, "
                   f"RSI Oversold={args.rsi_oversold}, Trend EMA={args.trend_ema}")
        
        logger.info(f"Enhanced parameters: Enabled={not args.no_adaptive}, "
                   f"Volatility Lookback={args.volatility_lookback}, "
                   f"Regime Lookback={args.regime_lookback}, "
                   f"Trailing Stop: {'Enabled' if not args.no_trailing_stop else 'Disabled'}, "
                   f"Stop %={args.trailing_stop}, "
                   f"Min Trade Duration={args.min_trade_duration}, "
                   f"Max Trades Per Day={args.max_trades_per_day}")
        
        # Create backtester instance
        backtester = Backtester(
            strategy=strategy,
            initial_capital=args.capital,
            commission_pct=args.commission,
            risk_per_trade_pct=args.risk
        )
        
        # Fetch historical data
        data = backtester.fetch_historical_data(
            exchange_id=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit
        )
        
        # Store symbol in dataframe attributes for strategy to access
        data.attrs['symbol'] = args.symbol
        
        # Check if we got any data
        if len(data) == 0:
            logger.error("No data available for backtesting. Check your date range and symbol.")
            return
        
        # Run backtest
        logger.info(f"Running backtest on {len(data)} candles from {data.index[0]} to {data.index[-1]}")
        results = backtester.run_backtest(data)
        
        # Plot results if not disabled
        if not args.no_plot:
            backtester.plot_results(results)
        
        # Log performance summary
        performance = results['performance']
        logger.info(f"\nPerformance Summary:")
        logger.info(f"Period: {data.index[0]} to {data.index[-1]}")
        logger.info(f"Strategy: {strategy.name}")
        logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
        logger.info(f"Initial Capital: ${args.capital:.2f}")
        logger.info(f"Final Capital: ${backtester.capital:.2f}")
        logger.info(f"Total Return: ${performance['total_return']:.2f} ({performance['return_pct']:.2f}%)")
        logger.info(f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%")
        logger.info(f"Number of Trades: {len(results['trades'])}")
        logger.info(f"Win Rate: {performance['win_rate']:.2f}%")
        logger.info(f"Average Profit: ${performance['avg_profit']:.2f}")
        logger.info(f"Average Loss: ${performance['avg_loss']:.2f}")
        logger.info(f"Profit Factor: {performance['profit_factor']:.2f}")
        logger.info(f"Maximum Drawdown: {performance['max_drawdown_pct']:.2f}%")
        
        logger.info("Backtester completed successfully")
        
    except Exception as e:
        logger.error(f"Error running backtester: {str(e)}")
        raise

if __name__ == "__main__":
    main()