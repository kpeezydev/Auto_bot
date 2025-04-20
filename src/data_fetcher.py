import ccxt
import pandas as pd
import logging
from typing import Dict, Any, Optional
import time

# Default values
TRADING_PAIR = "AVAX/USDT"
TIMEFRAME = "1h"
EXCHANGE = "binance"

# Configure logging
logger = logging.getLogger(__name__)

class DataFetcher:
    """
    Class for fetching market data from cryptocurrency exchanges using ccxt.
    """
    
    def __init__(self, exchange_id: str = EXCHANGE, api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None):
        """
        Initialize the DataFetcher with exchange connection.
        
        Args:
            exchange_id: The ID of the exchange to connect to (e.g., 'binance')
            api_key: API key for the exchange (optional for public data)
            api_secret: API secret for the exchange (optional for public data)
        """
        self.exchange_id = exchange_id
        
        # Initialize exchange connection
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
            logger.info(f"Successfully initialized connection to {exchange_id}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
            raise
    
    def fetch_ohlcv(self, symbol: str = TRADING_PAIR, timeframe: str = TIMEFRAME,
                    limit: int = 100, retries: int = 3, retry_delay: int = 5,
                    since: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data from the exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., 'AVAX/USDT')
            timeframe: Timeframe for the data (e.g., '1h', '1d')
            limit: Number of candles to fetch
            retries: Number of retry attempts if fetching fails
            retry_delay: Delay between retries in seconds
            since: Timestamp in milliseconds for start of data range
            
        Returns:
            DataFrame with OHLCV data
        """
        for attempt in range(retries):
            try:
                logger.info(f"Fetching {limit} {timeframe} candles for {symbol} from {self.exchange_id}")
                
                # Prepare parameters
                kwargs = {'limit': limit}
                if since:
                    kwargs['since'] = since
                
                # Fetch the OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"Successfully fetched {len(df)} candles")
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch OHLCV data after {retries} attempts")
                    raise
        
        # This should never be reached due to the raise in the loop
        return pd.DataFrame()
    
    def fetch_ticker(self, symbol: str = TRADING_PAIR) -> Dict[str, Any]:
        """
        Fetch current ticker information for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'AVAX/USDT')
            
        Returns:
            Dictionary with ticker information
        """
        logger.info(f"Fetching ticker for {symbol} from {self.exchange_id}")
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"Successfully fetched ticker for {symbol}")
            return ticker
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise
    
    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance information.
        
        Returns:
            Dictionary with balance information
        """
        logger.info(f"Fetching account balance from {self.exchange_id}")
        try:
            balance = self.exchange.fetch_balance()
            logger.info(f"Successfully fetched account balance")
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {str(e)}")
            raise

