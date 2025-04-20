import unittest
import pandas as pd
import numpy as np
from src.indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    """
    Test cases for the TechnicalIndicators class.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        dates = pd.date_range('20210101', periods=100)
        self.df = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_add_moving_averages(self):
        """Test adding moving averages."""
        df_with_ma = TechnicalIndicators.add_moving_averages(self.df, fast_period=10, slow_period=20)
        
        # Check that the new columns were added
        self.assertIn('sma_10', df_with_ma.columns)
        self.assertIn('sma_20', df_with_ma.columns)
        
        # Check that the values are reasonable
        # The SMA should be between the min and max of the close prices
        self.assertTrue(df_with_ma['sma_10'].min() >= self.df['close'].min())
        self.assertTrue(df_with_ma['sma_10'].max() <= self.df['close'].max())
    
    def test_add_rsi(self):
        """Test adding RSI."""
        df_with_rsi = TechnicalIndicators.add_rsi(self.df, period=14)
        
        # Check that the new column was added
        self.assertIn('rsi_14', df_with_rsi.columns)
        
        # Check that the values are in the expected range (0-100)
        self.assertTrue(df_with_rsi['rsi_14'].min() >= 0)
        self.assertTrue(df_with_rsi['rsi_14'].max() <= 100)
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators."""
        df_with_all = TechnicalIndicators.calculate_all_indicators(self.df)
        
        # Check that all expected columns were added
        expected_columns = [
            'sma_20', 'sma_50', 'rsi_14', 
            'macd', 'macd_signal', 'macd_histogram',
            'bb_lower', 'bb_middle', 'bb_upper'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df_with_all.columns)
        
        # Check that NaN values were dropped
        self.assertFalse(df_with_all.isnull().any().any())

if __name__ == '__main__':
    unittest.main()