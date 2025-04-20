import argparse
import logging
from src.utils import setup_logging, load_env_vars
from src.strategy import MovingAverageCrossStrategy
from src.trading_bot import TradingBot
from config.config import LOG_LEVEL, LOG_FILE, TRADING_PAIR, TIMEFRAME, EXCHANGE

def main():
    """
    Main function to run the trading bot.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default='paper',
                        help='Trading mode: backtest, paper, or live')
    parser.add_argument('--pair', type=str, default=TRADING_PAIR,
                        help=f'Trading pair (default: {TRADING_PAIR})')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME,
                        help=f'Timeframe for analysis (default: {TIMEFRAME})')
    parser.add_argument('--exchange', type=str, default=EXCHANGE,
                        help=f'Exchange to use (default: {EXCHANGE})')
    parser.add_argument('--strategy', type=str, default='ma_cross',
                        help='Trading strategy to use (default: ma_cross)')
    parser.add_argument('--run-once', action='store_true',
                        help='Run the bot once and exit')
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between trading cycles in minutes (default: 60)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    env_vars = load_env_vars()
    
    # Log startup information
    logger.info(f"Starting trading bot in {args.mode} mode")
    logger.info(f"Trading {args.pair} on {args.exchange} using {args.timeframe} timeframe")
    
    # Create strategy instance
    if args.strategy == 'ma_cross':
        strategy = MovingAverageCrossStrategy()
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    # Create trading bot instance
    paper_trading = args.mode == 'paper'
    
    if args.mode == 'backtest':
        # Import backtester
        from src.backtester import Backtester
        from src.data_fetcher import DataFetcher
        from src.indicators import TechnicalIndicators
        
        logger.info("Running backtest")
        
        # Create data fetcher
        data_fetcher = DataFetcher(
            exchange_id=args.exchange,
            api_key=env_vars.get('EXCHANGE_API_KEY'),
            api_secret=env_vars.get('EXCHANGE_SECRET_KEY')
        )
        
        # Fetch historical data
        df = data_fetcher.fetch_ohlcv(
            symbol=args.pair,
            timeframe=args.timeframe,
            limit=500  # Fetch enough data for a meaningful backtest
        )
        
        # Calculate indicators
        df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
        
        # Create backtester
        backtester = Backtester(strategy=strategy)
        
        # Run backtest
        results = backtester.run(df_with_indicators)
        
        # Plot results
        backtester.plot_results(results)
        
        # Print summary
        print("\nBacktest Summary:")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
    else:  # paper or live trading
        # Check if we have API keys
        if not env_vars.get('EXCHANGE_API_KEY') or not env_vars.get('EXCHANGE_SECRET_KEY'):
            logger.error("API keys not found. Please set EXCHANGE_API_KEY and EXCHANGE_SECRET_KEY in .env file")
            return
        
        # Create trading bot
        bot = TradingBot(
            strategy=strategy,
            exchange_id=args.exchange,
            trading_pair=args.pair,
            timeframe=args.timeframe,
            paper_trading=paper_trading
        )
        
        # Run the bot
        if args.run_once:
            logger.info("Running trading bot once")
            result = bot.run_once()
            logger.info(f"Trading cycle result: {result}")
        else:
            logger.info(f"Running trading bot every {args.interval} minutes")
            bot.run_scheduled(interval_minutes=args.interval)

if __name__ == "__main__":
    main()