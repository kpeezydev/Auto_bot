"""
Comprehensive Strategy Testing and Optimization Script
Systematically tests different parameter combinations to find optimal settings
"""

import logging
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime, timedelta
import json
import os

from src.strategy_enhanced_confluence import EnhancedConfluenceStrategy
from src.strategy_enhanced_adaptive import EnhancedAdaptiveStrategy
from src.strategy import MovingAverageCrossStrategy
from src.backtester import Backtester
from src.utils import setup_logging

# Configure logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

class StrategyOptimizer:
    """
    Comprehensive strategy optimization and testing framework.
    """
    
    def __init__(self, initial_capital=10000, commission_pct=0.1):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.results = []
        
    def test_single_strategy(self, strategy, symbol="BTC/USDT", timeframe="1h", 
                           days_back=365, exchange="binance"):
        """Test a single strategy configuration."""
        try:
            logger.info(f"Testing {strategy.name} on {symbol}")
            
            # Create backtester
            backtester = Backtester(
                strategy=strategy,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                risk_per_trade_pct=1.0
            )
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch data
            data = backtester.fetch_historical_data(
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=days_back * 24  # Approximate for hourly data
            )
            
            if len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} candles")
                return None
            
            # Run backtest
            results = backtester.run_backtest(data)
            performance = results['performance']
            
            # Extract key metrics
            metrics = {
                'strategy_name': strategy.name,
                'symbol': symbol,
                'timeframe': timeframe,
                'total_return_pct': performance['return_pct'],
                'win_rate': performance['win_rate'],
                'profit_factor': performance['profit_factor'],
                'max_drawdown_pct': performance['max_drawdown_pct'],
                'num_trades': len(results['trades']),
                'avg_profit': performance['avg_profit'],
                'avg_loss': performance['avg_loss'],
                'buy_hold_return': performance['buy_hold_return'],
                'alpha': performance['return_pct'] - performance['buy_hold_return'],
                'final_capital': backtester.capital,
                'data_points': len(data),
                'test_period_days': days_back
            }
            
            # Calculate additional metrics
            if len(results['trades']) > 0:
                winning_trades = [t for t in results['trades'] if t.pnl > 0]
                losing_trades = [t for t in results['trades'] if t.pnl < 0]
                
                metrics['largest_win'] = max([t.pnl for t in winning_trades]) if winning_trades else 0
                metrics['largest_loss'] = min([t.pnl for t in losing_trades]) if losing_trades else 0
                metrics['avg_trade_duration_hours'] = self._calculate_avg_duration(results['trades'])
            
            logger.info(f"Results: Return={metrics['total_return_pct']:.2f}%, "
                       f"Win Rate={metrics['win_rate']:.1f}%, "
                       f"Profit Factor={metrics['profit_factor']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error testing strategy: {str(e)}")
            return None
    
    def _calculate_avg_duration(self, trades):
        """Calculate average trade duration in hours."""
        durations = []
        for trade in trades:
            if hasattr(trade, 'exit_time') and hasattr(trade, 'entry_time'):
                duration = trade.exit_time - trade.entry_time
                hours = duration.total_seconds() / 3600
                durations.append(hours)
        return np.mean(durations) if durations else 0
    
    def optimize_enhanced_confluence(self, symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
        """
        Optimize Enhanced Confluence Strategy parameters.
        """
        logger.info("Starting Enhanced Confluence Strategy optimization...")
        
        # Parameter ranges to test
        param_grid = {
            'min_confluence_score': [0.5, 0.6, 0.7, 0.8],
            'min_adx_strength': [20.0, 25.0, 30.0],
            'min_risk_reward_ratio': [1.5, 2.0, 2.5, 3.0],
            'max_trades_per_day': [2, 3, 4, 5],
            'atr_stop_multiplier': [1.5, 2.0, 2.5]
        }
        
        best_results = {}
        
        for symbol in symbols:
            logger.info(f"\nOptimizing for {symbol}...")
            best_score = -999
            best_params = None
            best_metrics = None
            
            # Test all parameter combinations
            param_combinations = list(product(*param_grid.values()))
            total_combinations = len(param_combinations)
            
            logger.info(f"Testing {total_combinations} parameter combinations...")
            
            for i, params in enumerate(param_combinations):
                try:
                    # Create strategy with current parameters
                    strategy = EnhancedConfluenceStrategy(
                        min_confluence_score=params[0],
                        min_adx_strength=params[1],
                        min_risk_reward_ratio=params[2],
                        max_trades_per_day=params[3],
                        atr_stop_multiplier=params[4],
                        volume_confirmation=True,
                        use_dynamic_stops=True
                    )
                    
                    # Test strategy
                    metrics = self.test_single_strategy(strategy, symbol=symbol, days_back=180)
                    
                    if metrics is None:
                        continue
                    
                    # Calculate composite score (weighted combination of metrics)
                    score = self._calculate_composite_score(metrics)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'min_confluence_score': params[0],
                            'min_adx_strength': params[1],
                            'min_risk_reward_ratio': params[2],
                            'max_trades_per_day': params[3],
                            'atr_stop_multiplier': params[4]
                        }
                        best_metrics = metrics.copy()
                        best_metrics['composite_score'] = score
                        best_metrics['parameters'] = best_params
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Error testing combination {i}: {str(e)}")
                    continue
            
            best_results[symbol] = {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'best_score': best_score
            }
            
            logger.info(f"\nBest results for {symbol}:")
            logger.info(f"Parameters: {best_params}")
            logger.info(f"Return: {best_metrics['total_return_pct']:.2f}%")
            logger.info(f"Win Rate: {best_metrics['win_rate']:.1f}%")
            logger.info(f"Profit Factor: {best_metrics['profit_factor']:.2f}")
            logger.info(f"Composite Score: {best_score:.3f}")
        
        return best_results
    
    def _calculate_composite_score(self, metrics):
        """
        Calculate a composite score for ranking strategies.
        Weights different metrics based on importance.
        """
        try:
            # Ensure minimum trade count
            if metrics['num_trades'] < 10:
                return -999  # Penalize strategies with too few trades
            
            # Component scores (normalized to 0-1 range)
            return_score = min(metrics['total_return_pct'] / 100, 1.0)  # Cap at 100%
            win_rate_score = metrics['win_rate'] / 100
            profit_factor_score = min(metrics['profit_factor'] / 3.0, 1.0)  # Cap at 3.0
            drawdown_score = max(0, 1 - (metrics['max_drawdown_pct'] / 30))  # Penalize >30% DD
            alpha_score = max(0, min(metrics['alpha'] / 50, 1.0))  # Alpha vs buy&hold
            
            # Trade frequency score (prefer moderate frequency)
            trades_per_month = metrics['num_trades'] / (metrics['test_period_days'] / 30)
            if 2 <= trades_per_month <= 8:
                frequency_score = 1.0
            elif trades_per_month < 2:
                frequency_score = trades_per_month / 2
            else:
                frequency_score = max(0, 1 - (trades_per_month - 8) / 10)
            
            # Weighted composite score
            composite = (
                return_score * 0.25 +
                win_rate_score * 0.20 +
                profit_factor_score * 0.20 +
                drawdown_score * 0.15 +
                alpha_score * 0.10 +
                frequency_score * 0.10
            )
            
            return composite
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return -999
    
    def compare_strategies(self, symbols=["BTC/USDT", "ETH/USDT"]):
        """
        Compare different strategy implementations.
        """
        logger.info("Comparing different strategies...")
        
        strategies = {
            "Basic MA Cross": MovingAverageCrossStrategy(
                fast_ma_period=20,
                slow_ma_period=50,
                rsi_period=14
            ),
            "Enhanced Adaptive (Default)": EnhancedAdaptiveStrategy(
                fast_ma_period=12,
                slow_ma_period=26,
                use_adaptive_parameters=False,
                use_trailing_stop=True
            ),
            "Enhanced Adaptive (Enabled)": EnhancedAdaptiveStrategy(
                fast_ma_period=12,
                slow_ma_period=26,
                use_adaptive_parameters=True,
                use_trailing_stop=True
            ),
            "Enhanced Confluence (Conservative)": EnhancedConfluenceStrategy(
                min_confluence_score=0.7,
                min_adx_strength=30.0,
                min_risk_reward_ratio=2.5,
                max_trades_per_day=2
            ),
            "Enhanced Confluence (Balanced)": EnhancedConfluenceStrategy(
                min_confluence_score=0.6,
                min_adx_strength=25.0,
                min_risk_reward_ratio=2.0,
                max_trades_per_day=3
            ),
            "Enhanced Confluence (Aggressive)": EnhancedConfluenceStrategy(
                min_confluence_score=0.5,
                min_adx_strength=20.0,
                min_risk_reward_ratio=1.5,
                max_trades_per_day=4
            )
        }
        
        comparison_results = {}
        
        for symbol in symbols:
            logger.info(f"\nTesting strategies on {symbol}...")
            symbol_results = {}
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"Testing {strategy_name}...")
                metrics = self.test_single_strategy(strategy, symbol=symbol, days_back=365)
                
                if metrics:
                    metrics['composite_score'] = self._calculate_composite_score(metrics)
                    symbol_results[strategy_name] = metrics
            
            comparison_results[symbol] = symbol_results
        
        return comparison_results
    
    def save_results(self, results, filename="optimization_results.json"):
        """Save optimization results to file."""
        try:
            filepath = os.path.join("Sources/Auto_bot", filename)
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_comparison_summary(self, comparison_results):
        """Print a formatted summary of strategy comparison."""
        logger.info("\n" + "="*80)
        logger.info("STRATEGY COMPARISON SUMMARY")
        logger.info("="*80)
        
        for symbol, results in comparison_results.items():
            logger.info(f"\n{symbol} Results:")
            logger.info("-" * 50)
            
            # Sort by composite score
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['composite_score'], 
                                  reverse=True)
            
            for rank, (strategy_name, metrics) in enumerate(sorted_results, 1):
                logger.info(f"{rank}. {strategy_name}")
                logger.info(f"   Return: {metrics['total_return_pct']:.2f}% | "
                           f"Win Rate: {metrics['win_rate']:.1f}% | "
                           f"Profit Factor: {metrics['profit_factor']:.2f} | "
                           f"Max DD: {metrics['max_drawdown_pct']:.1f}% | "
                           f"Trades: {metrics['num_trades']} | "
                           f"Score: {metrics['composite_score']:.3f}")

def main():
    """Main optimization function."""
    try:
        logger.info("Starting comprehensive strategy testing and optimization...")
        
        optimizer = StrategyOptimizer(initial_capital=10000, commission_pct=0.1)
        
        # 1. Compare existing strategies
        logger.info("\n" + "="*60)
        logger.info("PHASE 1: STRATEGY COMPARISON")
        logger.info("="*60)
        
        comparison_results = optimizer.compare_strategies(
            symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        )
        
        optimizer.print_comparison_summary(comparison_results)
        optimizer.save_results(comparison_results, "strategy_comparison.json")
        
        # 2. Optimize Enhanced Confluence Strategy
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: PARAMETER OPTIMIZATION")
        logger.info("="*60)
        
        optimization_results = optimizer.optimize_enhanced_confluence(
            symbols=["BTC/USDT", "ETH/USDT"]
        )
        
        optimizer.save_results(optimization_results, "parameter_optimization.json")
        
        # 3. Print final recommendations
        logger.info("\n" + "="*60)
        logger.info("FINAL RECOMMENDATIONS")
        logger.info("="*60)
        
        for symbol, result in optimization_results.items():
            logger.info(f"\nOptimal settings for {symbol}:")
            logger.info(f"Parameters: {result['best_params']}")
            logger.info(f"Expected Return: {result['best_metrics']['total_return_pct']:.2f}%")
            logger.info(f"Expected Win Rate: {result['best_metrics']['win_rate']:.1f}%")
            logger.info(f"Expected Profit Factor: {result['best_metrics']['profit_factor']:.2f}")
        
        logger.info("\nOptimization completed successfully!")
        logger.info("Check the generated JSON files for detailed results.")
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        raise

if __name__ == "__main__":
    main()