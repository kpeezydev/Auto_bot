import logging
import argparse
from datetime import datetime, timedelta

from src.strategy import MovingAverageCrossStrategy
from src.strategy_enhanced import EnhancedMAStrategy
from src.backtester import Backtester
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run backtester for trading strategy")
    
    parser.add_argument("--exchange", type=str, default="binance",
                        help="Exchange to use (default: binance)")
    parser.add_argument("--symbol", type=str, default="AVAX/USDT",
                        help="Trading pair symbol (default: AVAX/USDT)")
    parser.add_argument("--timeframe", type=str, default="1h",
                        help="Timeframe for analysis (default: 1h)")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Number of candles to fetch (default: 1000)")
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
    parser.add_argument("--strategy", type=str, default="basic",
                        help="Strategy to use: 'basic' or 'enhanced' (default: enhanced)")
    
    return parser.parse_args()

def main():
    """Main function to run the backtester."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        logger.info("Starting backtester")
        
        # Create strategy instance
        if args.strategy == "basic":
            strategy = MovingAverageCrossStrategy()
            logger.info("Using basic Moving Average Cross Strategy")
        else:
            strategy = EnhancedMAStrategy()
            logger.info("Using Enhanced MA Strategy with Squeeze Pro indicator")
        
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
        print(f"Number of Trades: {performance['num_trades']}")
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