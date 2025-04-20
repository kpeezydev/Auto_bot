import pandas as pd
import pandas_ta as ta
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    Class for calculating technical indicators on OHLCV data.
    """
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pd.DataFrame:
        """
        Add Simple Moving Averages to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            
        Returns:
            DataFrame with added moving averages
        """
        try:
            logger.info(f"Calculating SMA with periods {fast_period} and {slow_period}")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate SMAs
            result_df[f'sma_{fast_period}'] = ta.sma(result_df['close'], length=fast_period)
            result_df[f'sma_{slow_period}'] = ta.sma(result_df['close'], length=slow_period)
            
            logger.info("Successfully calculated moving averages")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            raise
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with added RSI
        """
        try:
            logger.info(f"Calculating RSI with period {period}")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate RSI
            result_df[f'rsi_{period}'] = ta.rsi(result_df['close'], length=period)
            
            logger.info("Successfully calculated RSI")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast period for MACD calculation
            slow_period: Slow period for MACD calculation
            signal_period: Signal period for MACD calculation
            
        Returns:
            DataFrame with added MACD
        """
        try:
            logger.info(f"Calculating MACD with periods {fast_period}, {slow_period}, {signal_period}")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate MACD
            macd = ta.macd(result_df['close'], fast=fast_period, slow=slow_period, signal=signal_period)
            
            # Rename columns to be more intuitive
            macd.columns = [f'macd', f'macd_signal', f'macd_histogram']
            
            # Join with the original dataframe
            result_df = pd.concat([result_df, macd], axis=1)
            
            logger.info("Successfully calculated MACD")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for Bollinger Bands calculation
            std_dev: Number of standard deviations for the bands
            
        Returns:
            DataFrame with added Bollinger Bands
        """
        try:
            logger.info(f"Calculating Bollinger Bands with period {period} and std_dev {std_dev}")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate Bollinger Bands
            bbands = ta.bbands(result_df['close'], length=period, std=std_dev)
            
            # Rename columns to be more intuitive
            bbands.columns = ['bb_lower', 'bb_middle', 'bb_upper', 'bb_bandwidth', 'bb_percent']
            
            # Join with the original dataframe
            result_df = pd.concat([result_df, bbands], axis=1)
            
            logger.info("Successfully calculated Bollinger Bands")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Calculate all technical indicators based on configuration.
        
        Args:
            df: DataFrame with OHLCV data
            config: Dictionary with indicator parameters
            
        Returns:
            DataFrame with all indicators added
        """
        if config is None:
            # Default configuration
            config = {
                'sma_fast': 20,
                'sma_slow': 50,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std_dev': 2.0
            }
        
        try:
            logger.info("Calculating all technical indicators")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add all indicators
            result_df = TechnicalIndicators.add_moving_averages(
                result_df, 
                fast_period=config['sma_fast'], 
                slow_period=config['sma_slow']
            )
            
            result_df = TechnicalIndicators.add_rsi(
                result_df, 
                period=config['rsi_period']
            )
            
            result_df = TechnicalIndicators.add_macd(
                result_df, 
                fast_period=config['macd_fast'], 
                slow_period=config['macd_slow'], 
                signal_period=config['macd_signal']
            )
            
            result_df = TechnicalIndicators.add_bollinger_bands(
                result_df, 
                period=config['bb_period'], 
                std_dev=config['bb_std_dev']
            )
            
            # Drop NaN values that might have been introduced
            result_df = result_df.dropna()
            
            logger.info("Successfully calculated all indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create sample data
    import numpy as np
    dates = pd.date_range('20210101', periods=100)
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.default_rng().integers(1000, 10000, 100)
    }, index=dates)
    
    # Calculate indicators
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    # Print the result
    print(df_with_indicators.tail())