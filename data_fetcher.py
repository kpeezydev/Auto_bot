import ccxt
import pandas as pd
import time
import logging
from datetime import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
DEFAULT_TIMEFRAME = '1h'
DEFAULT_LIMIT = 100 # Number of candles to fetch

def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, limit: int = DEFAULT_LIMIT) -> pd.DataFrame | None:
    """
    Fetches OHLCV data for a given symbol and timeframe from an exchange.

    Args:
        exchange_id: The ID of the exchange (e.g., 'pionex', 'binance').
        symbol: The trading symbol (e.g., 'AVAX/USDT').
        timeframe: The timeframe ('1m', '5m', '1h', '1d', etc.).
        limit: The maximum number of candles to fetch.

    Returns:
        A pandas DataFrame with OHLCV data and datetime index, or None if an error occurs.
    """
    logging.info(f"Attempting to fetch {timeframe} OHLCV data for {symbol} from {exchange_id} (limit: {limit})...")
    try:
        # Initialize exchange (no API keys needed for public data)
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()

        # Check if the exchange supports fetching OHLCV data
        if not exchange.has['fetchOHLCV']:
            logging.error(f"Exchange {exchange_id} does not support fetchOHLCV.")
            return None

        # Fetch data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            logging.warning(f"No OHLCV data returned for {symbol} on {exchange_id}.")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert timestamp to datetime and set as index
        # CCXT returns timestamp in milliseconds
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        logging.info(f"Successfully fetched {len(df)} candles for {symbol} from {exchange_id}.")
        return df

    except ccxt.BadSymbol as e:
        logging.error(f"Error fetching OHLCV: {e}. Symbol '{symbol}' might be invalid for {exchange_id}.")
        return None
    except ccxt.NetworkError as e:
        logging.error(f"Network Error fetching OHLCV: {e}. Check connection.")
        # Optional: Implement retry logic here
        return None
    except ccxt.ExchangeError as e:
        logging.error(f"Exchange Error fetching OHLCV: {e}.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred fetching OHLCV: {e}")
        return None

# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Configuration for testing
    test_exchange = 'kucoin' # Change to your chosen exchange if needed
    test_symbol = 'AVAX/USDT'
    test_timeframe = '1h'
    test_limit = 5

    print(f"--- Running Data Fetcher Test ---")
    print(f"Exchange: {test_exchange}, Symbol: {test_symbol}, Timeframe: {test_timeframe}, Limit: {test_limit}")

    df_data = fetch_ohlcv(test_exchange, test_symbol, test_timeframe, test_limit)

    if df_data is not None:
        print("\nFetched Data Sample:")
        print(df_data.head())
        print("\nData Info:")
        df_data.info()
    else:
        print("\nFailed to fetch data.")

    print(f"--- End of Data Fetcher Test ---")