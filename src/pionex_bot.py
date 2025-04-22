import pandas as pd
import numpy as np
import logging
import time
import os
from typing import Dict, Any, Optional
from datetime import datetime
import ccxt
from dotenv import load_dotenv

# Assuming strategy and indicators are defined elsewhere (like in backtester)
from src.strategy import Strategy, SignalType # Adjust import if needed
from src.indicators import TechnicalIndicators # Adjust import if needed

# Load environment variables for API keys
load_dotenv(dotenv_path='../config/.env') # Adjust path if your .env is elsewhere

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("pionex_bot.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Pionex API Configuration ---
PIONEX_API_KEY = os.getenv('PIONEX_API_KEY')
PIONEX_SECRET_KEY = os.getenv('PIONEX_SECRET_KEY')

if not PIONEX_API_KEY or not PIONEX_SECRET_KEY:
    logger.error("Pionex API Key or Secret Key not found in environment variables.")
    raise ValueError("Pionex API credentials are required")

class PionexTradingBot:
    """
    Live trading bot for Pionex based on a defined strategy.
    """

    def __init__(self, strategy: Strategy, symbol: str, timeframe: str,
                 initial_capital: float = 1000.0, # Example starting capital (adjust as needed)
                 risk_per_trade_pct: float = 1.0,
                 leverage: int = 1): # Optional: Leverage for futures/margin
        """
        Initialize the Pionex Trading Bot.

        Args:
            strategy: Trading strategy instance.
            symbol: Trading symbol (e.g., 'BTC/USDT').
            timeframe: Candlestick timeframe (e.g., '1h', '15m').
            initial_capital: Starting capital (for reference/risk calculation).
            risk_per_trade_pct: Percentage of capital to risk per trade.
            leverage: Leverage to use (if applicable).
        """
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital # Note: Live capital managed by exchange
        self.risk_per_trade_pct = risk_per_trade_pct / 100.0
        self.leverage = leverage # Store leverage if needed for position sizing

        self.pionex = None # Placeholder for Pionex exchange instance
        self.current_position = None # Track open position {'id': str, 'symbol': str, 'direction': 'long'/'short', 'size': float, 'entry_price': float, 'stop_loss_price': float}
        self.ohlcv_data = pd.DataFrame() # Store recent OHLCV data
        self.market_info = None # Store market details (precision, limits)

        self._connect_to_pionex()
        self._load_market_info() # Load market info after connection

        logger.info(f"Initialized Pionex Trading Bot for {self.symbol} ({self.timeframe})")
        logger.info(f"Strategy: {strategy.name}, Risk per trade: {self.risk_per_trade_pct*100:.2f}%") # Corrected log

    def _connect_to_pionex(self):
        """Establish connection to the Pionex API."""
        try:
            # Note: Pionex might require specific headers or config options
            # Refer to ccxt documentation for Pionex: https://docs.ccxt.com/en/latest/manual.html#pionex
            # You might need to instantiate differently if using v2 or specific API endpoints
            self.pionex = ccxt.pionex({
                'apiKey': PIONEX_API_KEY,
                'secret': PIONEX_SECRET_KEY,
                # Add any other required options like 'headers' or specific API versions if needed
                # 'options': {
                #     'defaultType': 'spot', # or 'swap'/'future' if applicable
                # }
            })
            # Test connection (optional but recommended)
            self.pionex.load_markets()
            balance = self.pionex.fetch_balance() # Example API call
            logger.info("Successfully connected to Pionex API.")
            # logger.info(f"Account Balance: {balance}") # Be careful logging sensitive info
        except ccxt.AuthenticationError as e:
            logger.error(f"Pionex Authentication Error: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Pionex Exchange Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Pionex: {e}")
            raise

    def _load_market_info(self):
        """Load and store market information for the symbol."""
        if not self.pionex or not self.pionex.markets:
            logger.error("Cannot load market info: Exchange not connected or markets not loaded.")
            return
        try:
            self.market_info = self.pionex.market(self.symbol)
            logger.info(f"Loaded market info for {self.symbol}: Limits={self.market_info.get('limits')}, Precision={self.market_info.get('precision')}")
        except ccxt.ExchangeError as e:
            logger.error(f"Could not load market info for {self.symbol}: {e}")
            # Consider stopping the bot if market info is crucial
        except Exception as e:
            logger.error(f"Unexpected error loading market info: {e}")

    def fetch_latest_ohlcv(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch the latest OHLCV data from Pionex."""
        try:
            logger.debug(f"Fetching latest {limit} candles for {self.symbol} ({self.timeframe})")
            # fetch_ohlcv(symbol, timeframe='1m', since=None, limit=None, params={})
            ohlcv = self.pionex.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)

            if not ohlcv:
                logger.warning("No OHLCV data returned from Pionex.")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Combine with existing data if needed, ensuring no duplicates
            self.ohlcv_data = pd.concat([self.ohlcv_data, df]).drop_duplicates()
            # Keep only the most recent data (e.g., last 500 candles) to avoid memory issues
            self.ohlcv_data = self.ohlcv_data.iloc[-500:]

            logger.debug(f"Fetched {len(df)} new candles. Total data size: {len(self.ohlcv_data)}")
            return self.ohlcv_data # Return the updated dataframe

        except ccxt.NetworkError as e:
            logger.error(f"Pionex Network Error fetching OHLCV: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Pionex Exchange Error fetching OHLCV: {e}")
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
        return None # Return None on error

    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> Optional[float]:
        """Calculate position size based on risk and stop loss."""
        try:
            # Fetch current account balance (relevant asset, e.g., USDT)
            # Determine the quote currency (e.g., USDT from BTC/USDT)
            quote_currency = self.symbol.split('/')[-1]
            balance = self.get_account_balance(currency=quote_currency)
            if balance is None:
                logger.error(f"Could not retrieve account balance for {quote_currency} for position sizing.")
                return None
            if balance <= 0:
                logger.error(f"Insufficient balance ({balance} {quote_currency}) for trading.")
                return None

            logger.info(f"Available balance for risk calculation: {balance:.4f} {quote_currency}")
            risk_amount = balance * self.risk_per_trade_pct
            risk_per_unit = abs(entry_price - stop_loss_price)

            if risk_per_unit <= 0:
                logger.warning("Risk per unit is zero or negative. Cannot calculate position size.")
                return None

            position_size = risk_amount / risk_per_unit

            # Adjust for leverage if applicable (for futures/margin)
            # position_size *= self.leverage # Be careful with leverage

            # Apply market precision and limits
            if not self.market_info:
                logger.error("Market info not loaded, cannot apply precision/limits.")
                return None # Or handle differently

            min_size = self.market_info.get('limits', {}).get('amount', {}).get('min')
            # precision_amount = self.market_info.get('precision', {}).get('amount') # ccxt handles this internally with amount_to_precision

            if min_size is None:
                 logger.warning("Could not determine minimum order size from market info.")
                 # Proceed with caution or return None

            # Adjust size to exchange precision
            try:
                adjusted_size = self.pionex.amount_to_precision(self.symbol, position_size)
                logger.info(f"Raw size: {position_size}, Precision-adjusted size: {adjusted_size}")
                position_size = float(adjusted_size) # Convert back to float
            except Exception as prec_e:
                logger.error(f"Error applying amount precision: {prec_e}")
                return None


            # Check minimum order size
            if min_size is not None and position_size < min_size:
                logger.warning(f"Calculated size {position_size} is below minimum {min_size}. Cannot place trade.")
                return None

            # TODO: Check maximum order size if needed market_info['limits']['amount']['max']
            # TODO: Check cost limits market_info['limits']['cost']['min'] / ['max']
            # cost = position_size * entry_price
            # min_cost = self.market_info.get('limits', {}).get('cost', {}).get('min')
            # if min_cost is not None and cost < min_cost:
            #    logger.warning(f"Estimated cost {cost} is below minimum {min_cost}. Cannot place trade.")
            #    return None


            logger.info(f"Final calculated position size: {position_size:.8f} {self.symbol.split('/')[0]}") # Use more precision
            return position_size

        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error during position size calculation: {e}")
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return None

    def get_account_balance(self, currency: str = 'USDT') -> Optional[float]:
        """Fetch the free balance for a specific currency."""
        try:
            # Use fetch_balance() for a more complete view, then extract 'free'
            balance_data = self.pionex.fetch_balance()
            free_balance = balance_data.get('free', {}).get(currency)

            if free_balance is not None:
                logger.debug(f"Free balance for {currency}: {free_balance}")
                return float(free_balance)
            else:
                logger.warning(f"Currency '{currency}' not found in free balance details.")
                logger.debug(f"Full balance response: {balance_data}") # Log for debugging
                return 0.0 # Default to 0 if not found
        except ccxt.NetworkError as e:
            logger.error(f"Pionex Network Error fetching balance: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Pionex Exchange Error fetching balance: {e}")
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
        return None

    def generate_stop_loss(self, price: float, direction: str, atr_value: Optional[float] = None) -> float:
        """Generate stop loss price (similar to backtester)."""
        # This logic can be reused or adapted based on live trading needs
        if atr_value and atr_value > 0:
            stop_distance = 2 * atr_value # Example: 2 * ATR
        else:
            stop_distance = price * 0.02 # Example: 2% fixed stop
            logger.warning("ATR value not available or zero, using fixed 2% stop loss.")

        if direction == "long":
            return price - stop_distance
        else: # short
            return price + stop_distance

    def place_order(self, side: str, amount: float, price: Optional[float] = None, order_type: str = 'market', params={}) -> Optional[Dict]:
        """Place an order on Pionex."""
        try:
            logger.info(f"Attempting to place {order_type} {side} order for {amount:.8f} {self.symbol} at price {price if price else 'MARKET'}") # Log with precision

            # Ensure amount respects precision before placing order
            try:
                adjusted_amount = self.pionex.amount_to_precision(self.symbol, amount)
                amount = float(adjusted_amount) # Use the adjusted amount
            except Exception as prec_e:
                 logger.error(f"Error applying precision to order amount {amount}: {prec_e}")
                 return None

            # --- PIONEX API CALL ---
            # Refer to Pionex docs for specific params: https://pionex-doc.gitbook.io/apidocs/restful/orders/place-order
            # Example: Stop loss for spot might use 'stopPrice' and 'type': 'STOP_LIMIT' or similar
            # Example: Futures might have dedicated SL/TP params
            pionex_params = params.copy() # Start with any passed params
            # if order_type == 'market' and self.current_position and 'stop_loss' in self.current_position:
                 # This is tricky - market orders usually don't take SL directly.
                 # You might need a separate stop order AFTER the market order fills.
                 # Or use a LIMIT order with integrated stop features if Pionex supports it.
                 # pionex_params['stopPrice'] = self.current_position['stop_loss'] # Example, check Pionex docs

            logger.debug(f"Placing order with symbol={self.symbol}, type={order_type}, side={side}, amount={amount}, price={price}, params={pionex_params}")

            order = self.pionex.create_order(
                symbol=self.symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price, # Required for limit orders
                params=pionex_params
            )

            # Basic order validation after placement
            if order and 'id' in order:
                logger.info(f"Order placed successfully: ID={order['id']}, Status={order.get('status', 'N/A')}")
                logger.debug(f"Order details: {order}") # Log full details at debug level
                # TODO: Monitor order status until filled or cancelled if necessary, especially for limit orders.
                return order
            else:
                logger.error(f"Order placement failed or returned invalid response: {order}")
                return None

        except ccxt.InsufficientFunds as e:
            logger.error(f"Pionex Insufficient Funds: {e}")
        except ccxt.InvalidOrder as e:
            logger.error(f"Pionex Invalid Order: {e} (Check amount/price precision, limits)")
        except ccxt.NetworkError as e:
            logger.error(f"Pionex Network Error placing order: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Pionex Exchange Error placing order: {e}")
        except Exception as e:
            logger.error(f"Error placing order: {e}")
        return None

    def check_for_signals(self):
        """Fetch data, calculate indicators, generate signals, and manage trades."""
        latest_data = self.fetch_latest_ohlcv()

        if latest_data is None or latest_data.empty:
            logger.warning("No data available to check for signals.")
            return

        # Ensure enough data for indicators
        # min_periods = max(self.strategy.required_periods) # Assuming strategy has this attribute
        min_periods = 50 # Example: require at least 50 candles
        if len(latest_data) < min_periods:
            logger.info(f"Not enough data ({len(latest_data)}/{min_periods}) to generate reliable signals yet.")
            return

        # 1. Calculate Indicators
        try:
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(latest_data.copy()) # Use a copy
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return

        # 2. Generate Signals
        try:
            df_with_signals = self.strategy.generate_signals(df_with_indicators)
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return

        # 3. Process the latest signal
        last_row = df_with_signals.iloc[-1]
        current_signal = last_row.get('signal', SignalType.HOLD.value) # Default to HOLD
        current_price = last_row['close']
        current_time = last_row.name # Timestamp of the last candle

        logger.info(f"[{current_time}] Price: {current_price:.2f}, Signal: {SignalType(current_signal).name}")

        # --- Trade Logic ---
        # Sync position state before making decisions
        self.sync_position_state()

        # --- Trade Logic ---
        if self.current_position is None: # No open position according to our synced state
            if current_signal == SignalType.BUY.value:
                logger.info("BUY signal detected. Evaluating entry.")
                # Calculate SL and Size
                atr_value = last_row.get('atr_14') # Example ATR
                stop_loss_price = self.generate_stop_loss(current_price, "long", atr_value)
                position_size = self.calculate_position_size(current_price, stop_loss_price)

                if position_size:
                    # Place MARKET BUY order
                    # Place MARKET BUY order
                    # Consider using a LIMIT order slightly above current price to avoid bad fills in volatile markets
                    order = self.place_order(side='buy', amount=position_size, order_type='market')
                    if order and order.get('status') != 'rejected': # Check if order wasn't immediately rejected
                        # Assume filled for market order, but ideally confirm with fetch_order
                        filled_amount = order.get('filled', position_size if order.get('status') == 'closed' else 0) # Use filled if closed, else assume 0 until confirmed
                        avg_price = order.get('average', current_price) # Use average if available

                        if filled_amount > 0:
                            self.current_position = {
                                'id': order['id'],
                                'symbol': self.symbol,
                                'direction': 'long',
                                'size': filled_amount,
                                'entry_price': avg_price,
                                'stop_loss_price': stop_loss_price # Store intended SL
                            }
                            logger.info(f"LONG position opened (pending confirmation/update): {self.current_position}")
                            # TODO: Place separate STOP_LOSS order if needed, using order['id'] for reference
                        else:
                            logger.warning(f"Market BUY order {order['id']} placed but not confirmed filled immediately. Status: {order.get('status')}")
                            # Need logic to track this open order
                    else:
                         logger.error("Failed to place BUY order or order rejected.")
                else:
                    logger.warning("Could not calculate position size for BUY signal.")

            elif current_signal == SignalType.SELL.value: # Assuming SHORT selling allowed/configured
                logger.info("SELL signal detected (for potential SHORT). Evaluating entry.")
                # Calculate SL and Size
                atr_value = last_row.get('atr_14')
                stop_loss_price = self.generate_stop_loss(current_price, "short", atr_value)
                position_size = self.calculate_position_size(current_price, stop_loss_price)

                if position_size:
                     # Place MARKET SELL order (for SHORT)
                    # Place MARKET SELL order (for SHORT)
                    order = self.place_order(side='sell', amount=position_size, order_type='market')
                    if order and order.get('status') != 'rejected':
                        filled_amount = order.get('filled', position_size if order.get('status') == 'closed' else 0)
                        avg_price = order.get('average', current_price)

                        if filled_amount > 0:
                            self.current_position = {
                                'id': order['id'],
                                'symbol': self.symbol,
                                'direction': 'short',
                                'size': filled_amount,
                                'entry_price': avg_price,
                                'stop_loss_price': stop_loss_price
                            }
                            logger.info(f"SHORT position opened (pending confirmation/update): {self.current_position}")
                            # TODO: Place separate STOP_LOSS order
                        else:
                             logger.warning(f"Market SELL order {order['id']} placed but not confirmed filled immediately. Status: {order.get('status')}")
                             # Need logic to track this open order
                    else:
                        logger.error("Failed to place SELL order or order rejected.")
                else:
                    logger.warning("Could not calculate position size for SELL signal.")

        elif self.current_position is not None: # Manage open position (state confirmed by sync_position_state)
            direction = self.current_position['direction']
            position_size = self.current_position['size']

            # Check for Exit Signal
            exit_signal = False
            if direction == "long" and current_signal == SignalType.SELL.value:
                logger.info("SELL signal detected. Closing LONG position.")
                exit_signal = True
            elif direction == "short" and current_signal == SignalType.BUY.value:
                logger.info("BUY signal detected. Closing SHORT position.")
                exit_signal = True

            # Check for Stop Loss Hit - This is complex without exchange-side SL orders.
            # Relying solely on price crossing the SL level in the bot is risky due to latency.
            # Best practice: Place an actual Stop Loss order on the exchange when opening the position.
            # If using bot-side SL check:
            stop_hit = False
            sl_price = self.current_position.get('stop_loss_price')
            if sl_price:
                if direction == "long" and current_price <= sl_price:
                    logger.warning(f"Potential Stop Loss hit for LONG position (Price: {current_price} <= SL: {sl_price}). Closing.")
                    stop_hit = True
                elif direction == "short" and current_price >= sl_price:
                    logger.warning(f"Potential Stop Loss hit for SHORT position (Price: {current_price} >= SL: {sl_price}). Closing.")
                    stop_hit = True
            else:
                logger.warning("Stop loss price not found in current position state.")


            if exit_signal or stop_hit:
                logger.info(f"Attempting to close {direction.upper()} position of size {position_size} due to {'Exit Signal' if exit_signal else 'Stop Loss'}.")
                close_side = 'sell' if direction == 'long' else 'buy'
                # Ensure we close the exact position size
                order = self.place_order(side=close_side, amount=position_size, order_type='market')
                if order and order.get('status') != 'rejected':
                    # Assume closed, sync_position_state will confirm later
                    logger.info(f"Close order placed for {direction.upper()} position. ID: {order['id']}")
                    self.current_position = None # Optimistically reset state, sync will correct if needed
                else:
                    logger.error(f"Failed to place closing order for {direction.upper()} position. Manual intervention might be required.")
                    # Consider retry logic or alerting
            else:
                logger.info(f"Holding {direction.upper()} position. Size: {position_size}, Entry: {self.current_position.get('entry_price')}")

    def sync_position_state(self):
        """Fetch current position/order status from Pionex to sync internal state."""
        logger.debug("Syncing position state with exchange...")
        try:
            # Method 1: Fetch Positions (Primarily for Futures/Margin)
            # Check if the exchange and market type support fetch_positions
            if self.pionex.has.get('fetchPositions'):
                 # May need to specify symbol: self.pionex.fetch_positions([self.symbol])
                positions = self.pionex.fetch_positions()
                # Filter for the relevant symbol
                symbol_position = next((p for p in positions if p.get('symbol') == self.symbol and p.get('contracts', 0) > 0), None) # contracts might be 'size' or other field

                if symbol_position:
                    # Update self.current_position based on the fetched position data
                    # Important: Map the fields correctly (e.g., 'side', 'contracts', 'entryPrice')
                    size = float(symbol_position.get('contracts', 0)) # Adjust field name as needed
                    entry_price = float(symbol_position.get('entryPrice', 0)) # Adjust field name
                    direction = symbol_position.get('side') # 'long' or 'short'

                    # Basic check if fetched position seems valid
                    if size > 0 and direction:
                        stored_sl = self.current_position['stop_loss_price'] if self.current_position else None
                        self.current_position = {
                            'id': symbol_position.get('id', 'N/A'), # Position ID if available
                            'symbol': self.symbol,
                            'direction': direction,
                            'size': size,
                            'entry_price': entry_price,
                            'stop_loss_price': stored_sl # Keep bot's intended SL for now
                            # Add other relevant fields like leverage, margin, PNL if needed
                        }
                        logger.info(f"Synced position state from fetchPositions: {self.current_position}")
                        return # Position found and updated
                    else:
                         logger.debug(f"Ignoring fetched position data as size or direction is invalid: {symbol_position}")

                # If no position found via fetch_positions for the symbol
                if self.current_position:
                     logger.info("No active position found via fetchPositions, but bot state shows one. Resetting bot state.")
                     self.current_position = None
                else:
                     logger.debug("No active position found via fetchPositions, bot state is consistent.")
                return # Finished sync using fetchPositions

            # Method 2: Infer from Open Orders and Balance (More complex, fallback for Spot)
            # This is less reliable as it doesn't confirm filled positions directly.
            logger.warning("Exchange does not support fetchPositions. Position syncing might be incomplete (relying on bot's internal state).")
            # Basic check: If bot thinks it has a position, verify relevant balance exists.
            if self.current_position:
                 base_currency = self.symbol.split('/')[0]
                 balance = self.get_account_balance(currency=base_currency)
                 if balance is not None and balance < self.current_position['size'] * 0.9: # Check if balance roughly matches position size
                      logger.warning(f"Bot state shows position, but {base_currency} balance ({balance}) seems too low. Resetting state.")
                      self.current_position = None
                 else:
                      logger.info("Position state seems consistent with balance (basic check).")
            else:
                 logger.debug("No position in bot state.")


            # TODO: Could also check fetchOpenOrders to see if entry/exit orders are still pending.

        except ccxt.NetworkError as e:
            logger.error(f"Pionex Network Error syncing position state: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Pionex Exchange Error syncing position state: {e}")
        except Exception as e:
            logger.error(f"Error syncing position state: {e}", exc_info=True)


    def run(self, interval_seconds: int = 60):
        """Main loop for the trading bot."""
        logger.info("Starting trading bot loop...")
        while True:
            try:
                logger.info("-" * 30)
                # 1. Sync state with exchange
                self.sync_position_state() # Call sync here

                # 2. Check for signals and act
                self.check_for_signals()

                # 3. Log current state
                if self.current_position:
                    logger.info(f"Current State: Holding {self.current_position['direction']} {self.current_position['size']} {self.symbol}")
                else:
                    logger.info("Current State: No position open.")

                logger.info(f"--- Loop finished, sleeping for {interval_seconds} seconds ---")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Shutdown requested by user.")
                # TODO: Add cleanup logic (e.g., cancel open orders?)
                break
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
                # Implement backoff strategy for repeated errors
                logger.info("Waiting for 60 seconds before retrying...")
                time.sleep(60)


# --- Example Usage ---
if __name__ == "__main__":
    # Import a specific strategy
    from src.strategy import MovingAverageCrossStrategy # Example

    # --- Configuration ---
    BOT_SYMBOL = 'BTC/USDT'
    BOT_TIMEFRAME = '15m' # Check Pionex supported timeframes
    BOT_RISK_PCT = 1.0    # Risk 1% of capital per trade
    BOT_INTERVAL_SECONDS = 60 * 5 # Check signals every 5 minutes

    # --- Initialization ---
    try:
        # Create strategy instance
        strategy_instance = MovingAverageCrossStrategy(short_window=10, long_window=30) # Example params

        # Create bot instance
        bot = PionexTradingBot(
            strategy=strategy_instance,
            symbol=BOT_SYMBOL,
            timeframe=BOT_TIMEFRAME,
            risk_per_trade_pct=BOT_RISK_PCT
        )

        # --- Run the bot ---
        bot.run(interval_seconds=BOT_INTERVAL_SECONDS)

    except Exception as main_e:
        logger.critical(f"Failed to initialize or run the bot: {main_e}", exc_info=True)