import pandas as pd
from talipp.indicators import SMA
import numpy as np
import logging

# Import the data fetching function from data_fetcher.py
try:
    from data_fetcher import fetch_ohlcv
except ImportError:
    logging.error("Could not import fetch_ohlcv from data_fetcher.py. Make sure it's in the same directory.")
    def fetch_ohlcv(exchange_id, symbol, timeframe, limit): return None

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Strategy Configuration ---
FAST_MA_PERIOD = 10
SLOW_MA_PERIOD = 30

def apply_sma_crossover_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a Simple Moving Average (SMA) Crossover strategy to OHLCV data using talipp.

    Adds 'SMA_fast', 'SMA_slow', and 'signal' columns to the DataFrame.
    Signal values:
        1: Buy signal (Fast MA crosses above Slow MA)
       -1: Sell signal (Fast MA crosses below Slow MA)
        0: Hold signal (No crossover)

    Args:
        df: Pandas DataFrame with OHLCV data (must include 'close' column).

    Returns:
        The original DataFrame augmented with SMA and signal columns.
        Returns the original DataFrame unmodified if input is invalid or SMAs cannot be calculated.
    """
    if df is None or 'close' not in df.columns:
        logging.error("Invalid DataFrame received for strategy application.")
        return df

    if len(df) < SLOW_MA_PERIOD:
        logging.warning(f"DataFrame length ({len(df)}) is less than the slow MA period ({SLOW_MA_PERIOD}). Cannot calculate SMAs accurately.")
        df['SMA_fast'] = np.nan
        df['SMA_slow'] = np.nan
        df['signal'] = 0
        return df

    logging.info(f"Applying SMA Crossover Strategy using talipp (Fast: {FAST_MA_PERIOD}, Slow: {SLOW_MA_PERIOD})...")
    try:
        # --- Calculate SMAs using talipp ---
        close_prices = df['close'].tolist()
        sma_fast = SMA(period=FAST_MA_PERIOD)
        sma_slow = SMA(period=SLOW_MA_PERIOD)

        sma_fast_values = []
        sma_slow_values = []

        for price in close_prices:
            # Use the non-deprecated add() method
            # *** Assume add() returns the calculated value ***
            current_sma_fast = sma_fast.add(price)
            current_sma_slow = sma_slow.add(price)

            # Append result only if available (check if the returned value is not None)
            sma_fast_values.append(current_sma_fast if current_sma_fast is not None else np.nan)
            sma_slow_values.append(current_sma_slow if current_sma_slow is not None else np.nan)

        # Add calculated SMAs to the DataFrame
        df['SMA_fast'] = sma_fast_values
        df['SMA_slow'] = sma_slow_values

        # --- Signal Generation ---
        # Initialize signal column
        df['signal'] = 0
        sma_fast_col = 'SMA_fast'
        sma_slow_col = 'SMA_slow'

        # Find where Fast MA crosses above Slow MA (Buy Signal = 1)
        buy_condition = (df[sma_fast_col] > df[sma_slow_col]) & (df[sma_fast_col].shift(1) <= df[sma_slow_col].shift(1))
        df.loc[buy_condition, 'signal'] = 1

        # Find where Fast MA crosses below Slow MA (Sell Signal = -1)
        sell_condition = (df[sma_fast_col] < df[sma_slow_col]) & (df[sma_fast_col].shift(1) >= df[sma_slow_col].shift(1))
        df.loc[sell_condition, 'signal'] = -1

        # Fill initial NaN signals with 0
        df['signal'].fillna(0, inplace=True)
        # Ensure signal column is integer type
        df['signal'] = df['signal'].astype(int)

        logging.info("SMA Crossover Strategy applied successfully using talipp.")
        return df

    except Exception as e:
        logging.error(f"An error occurred during strategy application with talipp: {e}")
        # Return original DataFrame in case of error during calculation
        df['SMA_fast'] = np.nan
        df['SMA_slow'] = np.nan
        df['signal'] = 0
        return df


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Configuration for testing
    test_exchange = 'kucoin'
    test_symbol = 'AVAX/USDT'
    test_timeframe = '1h'
    test_limit = SLOW_MA_PERIOD + 50

    print(f"--- Running Strategy Test (using talipp) ---")
    print(f"Fetching data for {test_symbol} from {test_exchange}...")

    df_ohlcv = fetch_ohlcv(test_exchange, test_symbol, test_timeframe, test_limit)

    if df_ohlcv is not None:
        print("Data fetched successfully. Applying strategy...")
        df_strategy = apply_sma_crossover_strategy(df_ohlcv.copy())

        print("\nStrategy Applied. Displaying recent data with signals:")
        print(df_strategy.tail(15))

        signals_only = df_strategy[df_strategy['signal'] != 0]
        if not signals_only.empty:
            print("\nRows with generated signals:")
            print(signals_only)
        else:
            print("\nNo buy/sell signals generated in the fetched data range.")

    else:
        print("\nFailed to fetch data for strategy test.")

    print(f"--- End of Strategy Test ---")