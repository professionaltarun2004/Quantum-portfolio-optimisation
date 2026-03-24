import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketStateEngine:
    """
    Market State Engine (MSE) for Quantum Cognitive Portfolio System.
    Ingests price data to compute rolling metrics and classify the market regime.
    """
    
    def __init__(self, window_short=21, window_long=252):
        self.window_short = window_short
        self.window_long = window_long
    
    def compute_metrics(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates momentum, volatility, and drawdown for a portfolio/market proxy.
        If price_data has multiple columns, calculates the mean market proxy.
        """
        # Create a market proxy by taking the mean of all prices (equal weight proxy)
        if isinstance(price_data, pd.DataFrame) and len(price_data.columns) > 1:
            market_proxy = price_data.mean(axis=1)
        else:
            market_proxy = price_data.iloc[:, 0] if isinstance(price_data, pd.DataFrame) else price_data
            
        returns = market_proxy.pct_change().dropna()
        
        # Calculate trailing volatility (annualized)
        volatility = returns.rolling(window=self.window_short, min_periods=1).std() * np.sqrt(252)
        
        # Calculate momentum (short moving average vs long moving average)
        sma_short = market_proxy.rolling(window=self.window_short, min_periods=1).mean()
        sma_long = market_proxy.rolling(window=self.window_long, min_periods=1).mean()
        momentum = (sma_short / sma_long) - 1.0
        
        # Calculate Maximum Drawdown
        rolling_max = market_proxy.rolling(window=self.window_long, min_periods=1).max()
        drawdown = (market_proxy / rolling_max) - 1.0
        
        # Calculate recent return (5-day momentum equivalent)
        recent_return = market_proxy.pct_change(periods=5)
        
        metrics = pd.DataFrame({
            'volatility': volatility,
            'momentum': momentum,
            'drawdown': drawdown,
            'recent_return': recent_return
        }).dropna()
        
        return metrics

    def classify_regime(self, metrics: pd.Series) -> str:
        """
        Classifies the current regime based on recent metrics.
        Returns a discrete state string.
        """
        vol = metrics['volatility']
        mom = metrics['momentum']
        dd = metrics['drawdown']
        
        # Define thresholds
        high_vol_threshold = 0.20  # 20% annualized vol
        bear_drawdown_threshold = -0.15 # 15% drawdown
        
        if dd < bear_drawdown_threshold:
            return "BEAR_CRISIS"
        elif mom > 0 and vol < high_vol_threshold:
            return "BULL_STABLE"
        elif mom > 0 and vol >= high_vol_threshold:
            return "BULL_VOLATILE"
        elif mom <= 0 and vol < high_vol_threshold:
            return "BEAR_STABLE"
        else:
            return "BEAR_VOLATILE"
            
    def get_current_state(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process the data and return the final market state vector for the latest period.
        """
        if len(price_data) < self.window_long // 2:
            # Not enough data, return indifferent default state
            return {
                'regime': 'INSUFFICIENT_DATA',
                'metrics': {'volatility': 0.15, 'momentum': 0.0, 'drawdown': 0.0, 'recent_return': 0.0}
            }
            
        metrics_df = self.compute_metrics(price_data)
        if metrics_df.empty:
            return {
                'regime': 'INSUFFICIENT_DATA',
                'metrics': {'volatility': 0.15, 'momentum': 0.0, 'drawdown': 0.0, 'recent_return': 0.0}
            }
            
        latest_metrics = metrics_df.iloc[-1]
        regime = self.classify_regime(latest_metrics)
        
        return {
            'regime': regime,
            'metrics': latest_metrics.to_dict()
        }
