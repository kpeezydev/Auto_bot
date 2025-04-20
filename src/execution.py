import ccxt
import logging
import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from strategy import SignalType
from config.config import TRADING_PAIR, RISK_PER_TRADE

# Configure logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Enum for different types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Enum for order sides."""
    BUY = "buy"
    SELL = "sell"

class PositionType(Enum):
    """Enum for position types."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"

class ExecutionEngine:
    """
    Class for executing trades on cryptocurrency exchanges.
    """
    
    def __init__(self, exchange_id: str, api_key: str, api_secret: str, 
                 paper_trading: bool = True, risk_per_trade: float = RISK_PER_TRADE):
        """
        Initialize the execution engine.
        
        Args:
            exchange_id: ID of the exchange to use
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            paper_trading: Whether to use paper trading
            risk_per_trade: Percentage of capital to risk per trade (0.01 = 1%)
        """
        self.exchange_id = exchange_id
        self.paper_trading = paper_trading
        self.risk_per_trade = risk_per_trade
        self.current_position = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # Initialize exchange connection
        try:
            exchange_class = getattr(ccxt, exchange_id)
            
            # Set up exchange options
            options = {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            }
            
            # Add testnet/sandbox options if paper trading
            if paper_trading:
                if exchange_id == 'binance':
                    options['options'] = {'defaultType': 'future'}
                    options['urls'] = {'api': {'rest': 'https://testnet.binancefuture.com'}}
                elif exchange_id == 'bybit':
                    options['urls'] = {'api': {'rest': 'https://api-testnet.bybit.com'}}
                # Add more exchanges as needed
            
            self.exchange = exchange_class(options)
            
            logger.info(f"Successfully initialized connection to {exchange_id} "
                       f"({'Paper Trading' if paper_trading else 'Live Trading'})")
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
            raise
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance.
        
        Returns:
            Dictionary with balance information
        """
        try:
            logger.info(f"Fetching account balance from {self.exchange_id}")
            balance = self.exchange.fetch_balance()
            logger.info(f"Successfully fetched account balance")
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {str(e)}")
            raise
    
    def get_ticker(self, symbol: str = TRADING_PAIR) -> Dict[str, Any]:
        """
        Get current ticker information.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with ticker information
        """
        try:
            logger.info(f"Fetching ticker for {symbol} from {self.exchange_id}")
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Successfully fetched ticker for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise
    
    def calculate_position_size(self, price: float, stop_loss: float, 
                                available_balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current price
            stop_loss: Stop loss price
            available_balance: Available balance for trading
            
        Returns:
            Position size
        """
        try:
            # Calculate risk amount
            risk_amount = available_balance * self.risk_per_trade
            
            # Calculate risk per unit
            risk_per_unit = abs(price - stop_loss)
            
            if risk_per_unit <= 0:
                logger.warning("Risk per unit is zero or negative, using default position size")
                return available_balance * self.risk_per_trade / price
            
            # Calculate position size
            position_size = risk_amount / risk_per_unit
            
            logger.info(f"Calculated position size: {position_size} units "
                       f"(Risk: ${risk_amount:.2f}, Risk per unit: ${risk_per_unit:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            raise
    
    def place_order(self, symbol: str, order_type: OrderType, side: OrderSide, 
                    amount: float, price: Optional[float] = None, 
                    stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            symbol: Trading pair symbol
            order_type: Type of order
            side: Order side (buy/sell)
            amount: Amount to buy/sell
            price: Price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Dictionary with order information
        """
        try:
            logger.info(f"Placing {order_type.value} {side.value} order for {amount} {symbol} "
                       f"{'at price '+str(price) if price else ''} "
                       f"{'with stop price '+str(stop_price) if stop_price else ''}")
            
            # Prepare order parameters
            params = {}
            if stop_price is not None:
                params['stopPrice'] = stop_price
            
            # Place the order
            if order_type == OrderType.MARKET:
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type.value,
                    side=side.value,
                    amount=amount
                )
            else:
                if price is None:
                    raise ValueError("Price must be provided for non-market orders")
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type.value,
                    side=side.value,
                    amount=amount,
                    price=price,
                    params=params
                )
            
            logger.info(f"Successfully placed order: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            raise
    
    def get_open_position(self, symbol: str = TRADING_PAIR) -> Tuple[PositionType, float, float]:
        """
        Get information about the current open position.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (position_type, position_size, entry_price)
        """
        try:
            logger.info(f"Fetching open position for {symbol}")
            
            # For exchanges that support position fetching
            if hasattr(self.exchange, 'fetch_positions'):
                positions = self.exchange.fetch_positions([symbol])
                
                if positions and len(positions) > 0:
                    position = positions[0]
                    
                    if position['side'] == 'long' and position['contracts'] > 0:
                        return (PositionType.LONG, position['contracts'], position['entryPrice'])
                    elif position['side'] == 'short' and position['contracts'] > 0:
                        return (PositionType.SHORT, position['contracts'], position['entryPrice'])
            
            # If no position found or exchange doesn't support position fetching
            return (PositionType.NONE, 0.0, 0.0)
            
        except Exception as e:
            logger.error(f"Failed to fetch open position: {str(e)}")
            # Return no position in case of error
            return (PositionType.NONE, 0.0, 0.0)
    
    def execute_signal(self, signal: SignalType, symbol: str = TRADING_PAIR) -> Dict[str, Any]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with execution information
        """
        try:
            logger.info(f"Executing signal: {signal.value} for {symbol}")
            
            # Get current position
            self.current_position, self.position_size, self.entry_price = self.get_open_position(symbol)
            
            # Get current ticker
            ticker = self.get_ticker(symbol)
            current_price = ticker['last']
            
            # Get available balance
            balance = self.get_balance()
            
            # Extract the quote currency from the symbol (e.g., USDT from AVAX/USDT)
            quote_currency = symbol.split('/')[1]
            available_balance = balance['free'].get(quote_currency, 0.0)
            
            logger.info(f"Current position: {self.current_position.value}, Size: {self.position_size}, "
                       f"Entry price: {self.entry_price}, Current price: {current_price}, "
                       f"Available balance: {available_balance} {quote_currency}")
            
            # Execute signal based on current position
            result = {
                'signal': signal.value,
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': self.exchange.milliseconds(),
                'success': False,
                'order': None,
                'message': ""
            }
            
            # Handle BUY signal
            if signal == SignalType.BUY:
                if self.current_position == PositionType.NONE:
                    # Calculate stop loss (simple example: 2% below entry for long)
                    stop_loss = current_price * 0.98
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        price=current_price,
                        stop_loss=stop_loss,
                        available_balance=available_balance
                    )
                    
                    # Place market buy order
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        amount=position_size
                    )
                    
                    # Place stop loss order
                    stop_order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.STOP,
                        side=OrderSide.SELL,
                        amount=position_size,
                        price=stop_loss,
                        stop_price=stop_loss
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['stop_order'] = stop_order
                    result['message'] = f"Opened long position of {position_size} {symbol} at {current_price}"
                    
                elif self.current_position == PositionType.SHORT:
                    # Close existing short position
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,  # Buy to close short
                        amount=self.position_size
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['message'] = f"Closed short position of {self.position_size} {symbol} at {current_price}"
                
                else:  # Already in a long position
                    result['message'] = f"Already in a long position, no action taken"
            
            # Handle SELL signal
            elif signal == SignalType.SELL:
                if self.current_position == PositionType.NONE:
                    # Calculate stop loss (simple example: 2% above entry for short)
                    stop_loss = current_price * 1.02
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(
                        price=current_price,
                        stop_loss=stop_loss,
                        available_balance=available_balance
                    )
                    
                    # Place market sell order
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,
                        amount=position_size
                    )
                    
                    # Place stop loss order
                    stop_order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.STOP,
                        side=OrderSide.BUY,
                        amount=position_size,
                        price=stop_loss,
                        stop_price=stop_loss
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['stop_order'] = stop_order
                    result['message'] = f"Opened short position of {position_size} {symbol} at {current_price}"
                    
                elif self.current_position == PositionType.LONG:
                    # Close existing long position
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,  # Sell to close long
                        amount=self.position_size
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['message'] = f"Closed long position of {self.position_size} {symbol} at {current_price}"
                
                else:  # Already in a short position
                    result['message'] = f"Already in a short position, no action taken"
            
            # Handle CLOSE_LONG signal
            elif signal == SignalType.CLOSE_LONG:
                if self.current_position == PositionType.LONG:
                    # Close long position
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL,  # Sell to close long
                        amount=self.position_size
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['message'] = f"Closed long position of {self.position_size} {symbol} at {current_price}"
                else:
                    result['message'] = f"No long position to close"
            
            # Handle CLOSE_SHORT signal
            elif signal == SignalType.CLOSE_SHORT:
                if self.current_position == PositionType.SHORT:
                    # Close short position
                    order = self.place_order(
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,  # Buy to close short
                        amount=self.position_size
                    )
                    
                    result['success'] = True
                    result['order'] = order
                    result['message'] = f"Closed short position of {self.position_size} {symbol} at {current_price}"
                else:
                    result['message'] = f"No short position to close"
            
            # Handle HOLD signal
            elif signal == SignalType.HOLD:
                result['message'] = f"Hold signal received, no action taken"
            
            logger.info(result['message'])
            return result
            
        except Exception as e:
            logger.error(f"Error executing signal: {str(e)}")
            return {
                'signal': signal.value,
                'symbol': symbol,
                'timestamp': self.exchange.milliseconds() if hasattr(self, 'exchange') else None,
                'success': False,
                'message': f"Error: {str(e)}"
            }
