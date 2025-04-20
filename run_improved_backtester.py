import logging
import argparse
from datetime import datetime, timedelta

from src.strategy_improved import ImprovedMAStrategy
from src.backtester import Backtester
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtester with improved trading strategy")
    
    parser.add_argument("--exchange", type=str, default="binance",
                        help="Exchange to use (default: binance)")
    parser.add_argument("--symbol", type=str, default="AVAX/USDT",
                        help="Trading pair symbol (default: AVAX/USDT)")
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
    parser.add_argument("--start-date", type=str,
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help="Start date for backtest (format: YYYY-MM-DD, default: 365 days ago)")
    parser.add_argument("--end-date", type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help="End date for backtest (format: YYYY-MM-DD, default: today)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting of results")
    
    # Add parameters specific to the improved strategy
    parser.add_argument("--fast-ma", type=int, default=9,
                        help="Fast MA period (default: 9)")
    parser.add_argument("--slow-ma", type=int, default=21,
                        help="Slow MA period (default: 21)")
    parser.add_argument("--rsi-period", type=int, default=14,
                        help="RSI period (default: 14)")
    parser.add_argument("--rsi-overbought", type=int, default=65,
                        help="RSI overbought threshold (default: 65)")
    parser.add_argument("--rsi-oversold", type=int, default=35,
                        help="RSI oversold threshold (default: 35)")
    parser.add_argument("--trend-ema", type=int, default=100,
                        help="Trend EMA period (default: 100)")
    
    return parser.parse_args()

def main():
    """Main function to run the backtester with improved strategy."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        logger.info("Starting backtester with improved strategy")
        
        # Create improved strategy instance with parameters from command line
        strategy = ImprovedMAStrategy(
            fast_ma_period=args.fast_ma,
            slow_ma_period=args.slow_ma,
            rsi_period=args.rsi_period,
            rsi_overbought=args.rsi_overbought,
            rsi_oversold=args.rsi_oversold,
            trend_ema_period=args.trend_ema
        )
        
        logger.info(f"Using {strategy.name} with parameters: "
                   f"Fast MA={args.fast_ma}, Slow MA={args.slow_ma}, "
                   f"RSI Period={args.rsi_period}, RSI Overbought={args.rsi_overbought}, "
                   f"RSI Oversold={args.rsi_oversold}, Trend EMA={args.trend_ema}")
        
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
        
        # Print performance summary
        performance = results['performance']
        print(f"\nPerformance Summary:")
        print(f"Period: {data.index[0]} to {data.index[-1]}")
        print(f"Strategy: {strategy.name}")
        print(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")
        print(f"Initial Capital: ${args.capital:.2f}")
        print(f"Final Capital: ${backtester.capital:.2f}")
        print(f"Total Return: ${performance['total_return']:.2f} ({performance['return_pct']:.2f}%)")
        print(f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%")
        print(f"Number of Trades: {len(results['trades'])}")
        print(f"Win Rate: {performance['win_rate']:.2f}%")
        print(f"Average Profit: ${performance['avg_profit']:.2f}")
        print(f"Average Loss: ${performance['avg_loss']:.2f}")
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
        print(f"Maximum Drawdown: {performance['max_drawdown_pct']:.2f}%")
        
        logger.info("Backtester completed successfully")
        
    except Exception as e:
        logger.error(f"Error running backtester: {str(e)}")
        raise

if __name__ == "__main__":
    main()