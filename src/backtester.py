import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from src.strategy import Strategy, SignalType
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators

# Configure logging
logger = logging.getLogger(__name__)

class Trade:
    """Class to represent a single trade."""
    
    def __init__(self, entry_time: datetime, entry_price: float, direction: str, 
                 position_size: float, stop_loss: Optional[float] = None):
        """
        Initialize a trade.
        
        Args:
            entry_time: Time of trade entry
            entry_price: Price at entry
            direction: 'long' or 'short'
            position_size: Size of the position
            stop_loss: Stop loss price (optional)
        """
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.position_size = position_size
        self.stop_loss = stop_loss
        
        # These will be set when the trade is closed
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.status = "open"
    
    def close(self, exit_time: datetime, exit_price: float) -> None:
        """
        Close the trade.
        
        Args:
            exit_time: Time of trade exit
            exit_price: Price at exit
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = "closed"
        
        # Calculate profit/loss
        if self.direction == "long":
            self.pnl = (self.exit_price - self.entry_price) * self.position_size
            self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price * 100
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.position_size
            self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price * 100
            
        logger.info(f"Closed {self.direction} trade: Entry=${self.entry_price:.2f}, Exit=${self.exit_price:.2f}, PnL=${self.pnl:.2f} ({self.pnl_percent:.2f}%)")


class Backtester:
    """Class for backtesting trading strategies on historical data."""
    
    def __init__(self, strategy: Strategy, initial_capital: float = 10000.0, 
                 commission_pct: float = 0.1, risk_per_trade_pct: float = 1.0):
        """
        Initialize the backtester.
        
        Args:
            strategy: Trading strategy to backtest
            initial_capital: Initial capital for the backtest
            commission_pct: Commission percentage per trade
            risk_per_trade_pct: Percentage of capital to risk per trade
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100.0  # Convert to decimal
        self.risk_per_trade_pct = risk_per_trade_pct / 100.0  # Convert to decimal
        
        # Performance tracking
        self.capital = initial_capital
        self.equity_curve = []
        self.trades = []
        self.current_trade = None
        
        logger.info(f"Initialized backtester with {strategy.name} strategy")
        logger.info(f"Initial capital: ${initial_capital:.2f}, Commission: {commission_pct:.2f}%, Risk per trade: {risk_per_trade_pct:.2f}%")
    
    def fetch_historical_data(self, exchange_id: str, symbol: str, timeframe: str, 
                             start_date: Optional[str] = None, end_date: Optional[str] = None, 
                             limit: int = 1000) -> pd.DataFrame:
        try:
            # Set default period if not specified
            # Always use provided dates or defaults from arguments
            now = datetime.now()
            end_date = end_date if end_date else now.strftime('%Y-%m-%d')
            start_date = start_date if start_date else (now - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Validate dates are not in the future
            end_dt = pd.to_datetime(end_date)
            start_dt = pd.to_datetime(start_date)
            if end_dt > now:
                logger.warning(f"End date {end_date} is in the future, using current date instead")
                end_date = now.strftime('%Y-%m-%d')
                end_dt = now
            if start_dt > now:
                logger.warning(f"Start date {start_date} is in the future, using 365 days ago instead")
                start_date = (now - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                start_dt = now - pd.Timedelta(days=365)
            
            logger.info(f"Fetching historical data for {symbol} on {exchange_id} with {timeframe} timeframe")
            
            # Initialize data fetcher
            data_fetcher = DataFetcher(exchange_id=exchange_id)
            
            # Fetch OHLCV data iteratively to cover the date range
            all_data = pd.DataFrame()
            current_end_date = pd.to_datetime(end_date) if end_date else datetime.now()
            start_dt = pd.to_datetime(start_date) if start_date else None

            logger.info(f"Attempting to fetch data from {start_dt if start_dt else 'beginning'} to {current_end_date}")

            while True:
                logger.info(f"Fetching chunk ending at {current_end_date} with limit {limit}")
                # Fetch data ending at current_end_date
                # Note: ccxt fetch_ohlcv 'since' is a timestamp in milliseconds
                since_timestamp = int(current_end_date.timestamp() * 1000) if current_end_date else None

                chunk_df = data_fetcher.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=since_timestamp - (limit * data_fetcher.exchange.parse_timeframe(timeframe) * 1000) # Estimate start time for chunk
                )

                if chunk_df.empty:
                    logger.warning(f"No data fetched for chunk ending at {current_end_date}. Possible causes:")
                    logger.warning("- Exchange may not have data for this period")
                    logger.warning("- API rate limits may be reached")
                    logger.warning("- Invalid parameters (symbol/timeframe)")
                    break

                # Append chunk data, removing duplicates if any
                all_data = pd.concat([chunk_df, all_data]).drop_duplicates().sort_index()

                # Determine the earliest date in the fetched chunk
                earliest_date_in_chunk = chunk_df.index.min()

                # If we have reached or gone past the desired start date, or if the chunk is smaller than the limit
                # (indicating no more historical data is available before this chunk), stop fetching.
                if (start_dt and earliest_date_in_chunk <= start_dt) or len(chunk_df) < limit:
                    logger.info("Reached desired start date or end of available data.")
                    break

                # Set the end date for the next chunk to the earliest date of the current chunk
                current_end_date = earliest_date_in_chunk

            # Filter by date range after fetching all data
            if start_dt:
                all_data = all_data[all_data.index >= start_dt]
            if end_date:
                all_data = all_data[all_data.index <= pd.to_datetime(end_date)]

            if all_data.empty:
                 logger.warning("No data available for the specified date range.")
                 return pd.DataFrame()

            logger.info(f"Fetched total {len(all_data)} candles from {all_data.index[0]} to {all_data.index[-1]}")

            return all_data

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def calculate_position_size(self, price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current price
            stop_loss: Stop loss price
            
        Returns:
            Position size
        """
        risk_amount = self.capital * self.risk_per_trade_pct
        risk_per_unit = abs(price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.warning("Risk per unit is zero or negative, using default position size")
            return risk_amount / price
        
        position_size = risk_amount / risk_per_unit
        return position_size
    
    def generate_stop_loss(self, price: float, direction: str, atr_value: Optional[float] = None) -> float:
        """
        Generate a stop loss price based on ATR or fixed percentage.
        
        Args:
            price: Current price
            direction: Trade direction (f"long' or 'short')
            atr_value: ATR value if available
            
        Returns:
            Stop loss price
        """
        if atr_value and atr_value > 0:
            # Use ATR-based stop loss (2 * ATR)
            stop_distance = 2 * atr_value
        else:
            # Use fixed percentage (2%)
            stop_distance = price * 0.02
        
        if direction == "long":
            return price - stop_distance
        else:  # short
            return price + stop_distance
    
    def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a backtest on the provided data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info(f"Starting backtest with {self.strategy.name} strategy")
            
            # Reset performance tracking
            self.capital = self.initial_capital
            self.equity_curve = [self.initial_capital]
            self.trades = []
            self.current_trade = None
            
            # Calculate indicators
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            
            # Generate signals
            df_with_signals = self.strategy.generate_signals(df_with_indicators)
            
            # Iterate through each candle
            for i in range(1, len(df_with_signals)):
                current_row = df_with_signals.iloc[i]
                
                current_time = current_row.name
                current_price = current_row['close']
                current_signal = current_row['signal']
                
                # Check for ATR if available for stop loss calculation
                atr_value = current_row.get('atr_14', None)
                
                # Process signals
                if self.current_trade is None:  # No open position
                    if current_signal == SignalType.BUY.value:
                        # Generate stop loss
                        stop_loss = self.generate_stop_loss(current_price, "long", atr_value)
                        
                        # Calculate position size
                        position_size = self.calculate_position_size(current_price, stop_loss)
                        
                        # Open long position
                        self.current_trade = Trade(
                            entry_time=current_time,
                            entry_price=current_price,
                            direction="long",
                            position_size=position_size,
                            stop_loss=stop_loss
                        )
                        
                        # Apply commission
                        commission = current_price * position_size * self.commission_pct
                        self.capital -= commission
                        
                        logger.info(f"Opened LONG position at ${current_price:.2f}, Stop Loss: ${stop_loss:.2f}, Size: {position_size:.6f}")
                        
                    elif current_signal == SignalType.SELL.value:
                        # Generate stop loss
                        stop_loss = self.generate_stop_loss(current_price, "short", atr_value)
                        
                        # Calculate position size
                        position_size = self.calculate_position_size(current_price, stop_loss)
                        
                        # Open short position
                        self.current_trade = Trade(
                            entry_time=current_time,
                            entry_price=current_price,
                            direction="short",
                            position_size=position_size,
                            stop_loss=stop_loss
                        )
                        
                        # Apply commission
                        commission = current_price * position_size * self.commission_pct
                        self.capital -= commission
                        
                        logger.info(f"Opened SHORT position at ${current_price:.2f}, Stop Loss: ${stop_loss:.2f}, Size: {position_size:.6f}")
                
                elif self.current_trade.status == "open":  # Manage open position
                    # Check for stop loss hit
                    stop_hit = False
                    if self.current_trade.direction == "long" and current_row['low'] <= self.current_trade.stop_loss:
                        stop_hit = True
                        exit_price = self.current_trade.stop_loss  # Assume filled at stop price
                    elif self.current_trade.direction == "short" and current_row['high'] >= self.current_trade.stop_loss:
                        stop_hit = True
                        exit_price = self.current_trade.stop_loss  # Assume filled at stop price
                    
                    # Check for exit signal
                    exit_signal = False
                    if self.current_trade.direction == "long" and current_signal == SignalType.SELL.value:
                        exit_signal = True
                        exit_price = current_price
                    elif self.current_trade.direction == "short" and current_signal == SignalType.BUY.value:
                        exit_signal = True
                        exit_price = current_price
                    
                    # Close position if stop hit or exit signal
                    if stop_hit or exit_signal:
                        # Close the trade
                        self.current_trade.close(current_time, exit_price)
                        
                        # Apply commission
                        commission = exit_price * self.current_trade.position_size * self.commission_pct
                        self.capital -= commission
                        
                        # Update capital
                        self.capital += self.current_trade.pnl
                        
                        # Add to trades list
                        self.trades.append(self.current_trade)
                        
                        # Reset current trade
                        self.current_trade = None
                        
                        reason = "Stop Loss" if stop_hit else "Exit Signal"
                        logger.info(f"Closed trade due to {reason}. New capital: ${self.capital:.2f}")
                
                # Update equity curve
                if self.current_trade is not None and self.current_trade.status == "open":
                    # Calculate unrealized PnL
                    if self.current_trade.direction == "long":
                        unrealized_pnl = (current_price - self.current_trade.entry_price) * self.current_trade.position_size
                    else:  # short
                        unrealized_pnl = (self.current_trade.entry_price - current_price) * self.current_trade.position_size
                    
                    # Add to equity curve
                    self.equity_curve.append(self.capital + unrealized_pnl)
                else:
                    # Add to equity curve
                    self.equity_curve.append(self.capital)
            
            # Close any open trades at the end of the backtest
            if self.current_trade is not None and self.current_trade.status == "open":
                last_price = df_with_signals.iloc[-1]['close']
                last_time = df_with_signals.index[-1]
                
                # Close the trade
                self.current_trade.close(last_time, last_price)
                
                # Apply commission
                commission = last_price * self.current_trade.position_size * self.commission_pct
                self.capital -= commission
                
                # Update capital
                self.capital += self.current_trade.pnl
                
                # Add to trades list
                self.trades.append(self.current_trade)
                
                # Reset current trade
                self.current_trade = None
                
                logger.info(f"Closed final trade at end of backtest. Final capital: ${self.capital:.2f}")
            
            # Calculate performance metrics
            performance = self.calculate_performance_metrics()
            
            logger.info(f"Backtest completed. Final capital: ${self.capital:.2f}, Return: {performance['return_pct']:.2f}%")
            
            return {
                'equity_curve': self.equity_curve,
                'trades': self.trades,
                'performance': performance,
                'data': df_with_signals
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics from the backtest results.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Basic metrics
            total_return = self.capital - self.initial_capital
            return_pct = (total_return / self.initial_capital) * 100
            
            # Calculate buy & hold return
            buy_hold_return = 0
            if self.trades:
                first_price = self.trades[0].entry_price
                last_price = self.trades[-1].exit_price
                if first_price and last_price:
                    buy_hold_return = (last_price - first_price) / first_price * 100
            
            # Trade metrics
            num_trades = len(self.trades)
            if num_trades == 0:
                return {
                    'total_return': total_return,
                    'return_pct': return_pct,
                    'buy_hold_return': buy_hold_return,
                    'num_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown_pct': 0
                }
            
            winning_trades = [trade for trade in self.trades if trade.pnl > 0]
            losing_trades = [trade for trade in self.trades if trade.pnl <= 0]
            
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            win_rate = (num_winning / num_trades) * 100
            
            # Average profit/loss
            avg_profit = sum(trade.pnl for trade in winning_trades) / num_winning if num_winning > 0 else 0
            avg_loss = sum(trade.pnl for trade in losing_trades) / num_losing if num_losing > 0 else 0
            
            # Profit factor
            gross_profit = sum(trade.pnl for trade in winning_trades)
            gross_loss = abs(sum(trade.pnl for trade in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Maximum drawdown
            equity_array = np.array(self.equity_curve)
            max_equity = np.maximum.accumulate(equity_array)
            drawdown = (max_equity - equity_array) / max_equity * 100
            max_drawdown_pct = np.max(drawdown) if len(drawdown) > 0 else 0
            
            return {
                'total_return': total_return,
                'return_pct': return_pct,
                'buy_hold_return': buy_hold_return,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown_pct': max_drawdown_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def plot_results(self, results: Dict[str, Any], show_trades: bool = True) -> None:
        """
        Plot the backtest results.
        
        Args:
            results: Dictionary with backtest results
            show_trades: Whether to show trade markers on the chart
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Get data
            df = results['data']
            equity_curve = results['equity_curve']
            trades = results['trades']
            
            # Print performance summary
            performance = results['performance']
            print("\nPerformance Summary:")
            print(f"Total Return: ${performance['total_return']:.2f} ({performance['return_pct']:.2f}%)")
            print(f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%")
            print(f"Number of Trades: {performance['num_trades']}")
            print(f"Win Rate: {performance['win_rate']:.2f}%")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
            # Plot price chart
            ax1.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)
            
            # Plot moving averages if available
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                ax1.plot(df.index, df['sma_20'], label='SMA 20', color='orange', alpha=0.7)
                ax1.plot(df.index, df['sma_50'], label='SMA 50', color='red', alpha=0.7)
            
            # Plot trade markers if requested
            if show_trades and trades:
                # Separate trades by direction
                long_trades = [t for t in trades if t.direction == 'long']
                short_trades = [t for t in trades if t.direction == 'short']
                
                # Plot entries and exits
                if long_trades:
                    ax1.scatter([t.entry_time for t in long_trades], 
                               [t.entry_price for t in long_trades], 
                               marker='^', color='green', s=100, label='Long Entry')
                    ax1.scatter([t.exit_time for t in long_trades], 
                               [t.exit_price for t in long_trades], 
                               marker='x', color='black', s=100, label='Exit')
                
                if short_trades:
                    ax1.scatter([t.entry_time for t in short_trades], 
                               [t.entry_price for t in short_trades], 
                               marker='v', color='red', s=100, label='Short Entry')
                    ax1.scatter([t.exit_time for t in short_trades], 
                               [t.exit_price for t in short_trades], 
                               marker='x', color='black', s=100)
            
            # Set up price chart
            ax1.set_title('Backtest Results: Price Chart with Trades')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot equity curve
            ax2.plot(df.index, equity_curve, label='Equity Curve', color='green')
            ax2.set_title('Equity Curve')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Equity')
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis dates
            fig.autofmt_xdate()
            
            # Add performance metrics as text
            performance = results['performance']
            metrics_text = (
                f"Total Return: ${performance['total_return']:.2f} ({performance['return_pct']:.2f}%)\n"
                f"Buy & Hold Return: {performance['buy_hold_return']:.2f}%\n"
                f"Number of Trades: {performance['num_trades']}\n"
                f"Win Rate: {performance['win_rate']:.2f}%\n"
                f"Profit Factor: {performance['profit_factor']:.2f}\n"
                f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%"
            )
            
            # Add text box with metrics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.02, 0.97, metrics_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import strategy
    from strategy import MovingAverageCrossStrategy
    
    # Create strategy instance
    strategy = MovingAverageCrossStrategy()
    
    # Create backtester instance
    backtester = Backtester(strategy=strategy)
    
    # Fetch historical data
