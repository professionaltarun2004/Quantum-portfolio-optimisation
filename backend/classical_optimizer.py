import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import time

class ClassicalOptimizer:
    """Classical portfolio optimization using mean-variance and ML techniques."""
    
    def __init__(self):
        self.method = "mean_variance"
        self.results = {}
    
    def optimize(self, returns_data, risk_tolerance=1.0, method="mean_variance"):
        """
        Optimize portfolio using classical methods.
        
        Args:
            returns_data: DataFrame of asset returns
            risk_tolerance: Risk tolerance parameter (higher = more risk)
            method: Optimization method ('mean_variance', 'ml_enhanced')
        """
        start_time = time.time()
        
        # Validate input data
        if returns_data.empty or len(returns_data) < 2:
            raise ValueError("Insufficient data for optimization")
        
        # Calculate expected returns and covariance
        expected_returns = returns_data.mean().values
        
        # Use Ledoit-Wolf shrinkage for better covariance estimation
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_data).covariance_
        
        n_assets = len(expected_returns)
        
        if method == "mean_variance":
            weights = self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
        elif method == "lasso":
            weights = self._lasso_optimization(returns_data, expected_returns, cov_matrix, risk_tolerance)
        elif method == "ridge":
            weights = self._ridge_optimization(returns_data, expected_returns, cov_matrix, risk_tolerance)
        elif method == "genetic_algorithm":
            weights = self._genetic_algorithm_optimization(expected_returns, cov_matrix, risk_tolerance)
        elif method == "neural_network":
            weights = self._neural_network_optimization(returns_data, expected_returns, cov_matrix, risk_tolerance)
        else:
            weights = self._ml_enhanced_optimization(returns_data, expected_returns, cov_matrix, risk_tolerance)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns) * 252  # Annualized
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        computation_time = time.time() - start_time
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'computation_time': computation_time,
            'method': method,
            'tickers': returns_data.columns.tolist(),
            'convergence_history': getattr(self, 'convergence_history', [])
        }
    
    def _mean_variance_optimization(self, expected_returns, cov_matrix, risk_tolerance):
        """Standard Markowitz mean-variance optimization using scipy."""
        n_assets = len(expected_returns)
        
        # Define objective function (minimize negative utility)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Utility = return - risk_penalty * risk^2
            utility = portfolio_return - (1/risk_tolerance) * portfolio_risk**2
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds: long-only positions, max 40% in any asset
        bounds = [(0, 0.4) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Solve optimization
        try:
            result = minimize(
                objective, 
                initial_weights, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success and result.x is not None:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
        except Exception:
            # Fallback to equal weights
            return np.ones(n_assets) / n_assets
    
    def _ml_enhanced_optimization(self, returns_data, expected_returns, cov_matrix, risk_tolerance):
        """ML-enhanced optimization with simplified feature engineering."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Simplified approach: use rolling statistics as features
            enhanced_returns = expected_returns.copy()
            
            if len(returns_data) > 30:  # Need sufficient data
                for i, ticker in enumerate(returns_data.columns):
                    try:
                        ticker_returns = returns_data[ticker].dropna()
                        
                        if len(ticker_returns) < 20:
                            continue
                        
                        # Create simple features: rolling means and volatility
                        prices = (1 + ticker_returns).cumprod()
                        sma_5 = prices.rolling(5, min_periods=1).mean()
                        sma_10 = prices.rolling(10, min_periods=1).mean()
                        vol_5 = ticker_returns.rolling(5, min_periods=1).std()
                        
                        # Align all data to same length
                        min_len = min(len(sma_5), len(sma_10), len(vol_5), len(ticker_returns))
                        
                        # Create feature matrix
                        features = np.column_stack([
                            sma_5.iloc[:min_len].ffill().bfill(),
                            sma_10.iloc[:min_len].ffill().bfill(),
                            vol_5.iloc[:min_len].ffill().bfill()
                        ])
                        
                        # Target: next period returns (shift by 1)
                        targets = ticker_returns.iloc[:min_len].values
                        
                        # Use lagged features to predict returns
                        if min_len > 15:
                            X = features[:-1]  # Features: t-1
                            y = targets[1:]    # Target: t (next period)
                            
                            # Remove any NaN values
                            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                            X_clean = X[valid_mask]
                            y_clean = y[valid_mask]
                            
                            if len(X_clean) > 10 and len(y_clean) > 10:
                                # Train model
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X_clean)
                                
                                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                                rf_model.fit(X_scaled, y_clean)
                                
                                # Predict using last available features
                                last_features = features[-1:].reshape(1, -1)
                                if not np.isnan(last_features).any():
                                    last_scaled = scaler.transform(last_features)
                                    prediction = rf_model.predict(last_scaled)[0]
                                    
                                    # Blend with historical mean (conservative approach)
                                    historical_mean = ticker_returns.mean()
                                    enhanced_returns[i] = 0.7 * historical_mean + 0.3 * prediction
                        
                    except Exception as e:
                        # If ML prediction fails for this ticker, keep original expected return
                        continue
            
            # Use enhanced returns in mean-variance optimization
            return self._mean_variance_optimization(enhanced_returns, cov_matrix, risk_tolerance)
            
        except Exception as e:
            # If entire ML enhancement fails, fallback to standard optimization
            return self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
    
    def _create_features(self, returns_data):
        """Create technical features for ML model."""
        features = {}
        
        for ticker in returns_data.columns:
            prices = (1 + returns_data[ticker]).cumprod()
            
            # Technical indicators
            sma_5 = prices.rolling(5).mean()
            sma_20 = prices.rolling(20).mean()
            rsi = self._calculate_rsi(prices)
            volatility = returns_data[ticker].rolling(20).std()
            
            # Combine features and handle NaN values
            try:
                # Fill NaN values
                sma_5_filled = sma_5.bfill().ffill()
                sma_20_filled = sma_20.bfill().ffill()
                rsi_filled = rsi.fillna(50)
                vol_filled = volatility.fillna(volatility.mean())
                
                # Ensure all series have the same length
                min_length = min(len(sma_5_filled), len(sma_20_filled), 
                               len(rsi_filled), len(vol_filled))
                
                feature_array = np.column_stack([
                    sma_5_filled.iloc[:min_length],
                    sma_20_filled.iloc[:min_length], 
                    rsi_filled.iloc[:min_length],
                    vol_filled.iloc[:min_length]
                ])
                
                # Remove any remaining NaN rows
                feature_array = feature_array[~np.isnan(feature_array).any(axis=1)]
                
                features[ticker] = feature_array
                
            except Exception as e:
                # Fallback: create simple features
                simple_features = np.ones((len(prices), 4)) * prices.mean()
                features[ticker] = simple_features
        
        return features
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _lasso_optimization(self, returns_data, expected_returns, cov_matrix, risk_tolerance):
        """Lasso regression for portfolio optimization with L1 regularization."""
        from sklearn.linear_model import Lasso
        
        try:
            n_assets = len(expected_returns)
            
            # Use Lasso for feature selection
            # Create target variable (future returns)
            if len(returns_data) > 30:
                X = returns_data.iloc[:-1].values  # Features: lagged returns
                y = returns_data.mean().values     # Target: expected returns
                
                # Fit Lasso model
                lasso = Lasso(alpha=0.01, positive=True)  # Positive weights only
                lasso.fit(X.T, y)  # Transpose to have assets as samples
                
                # Get weights from Lasso coefficients
                weights = np.abs(lasso.coef_)
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_assets) / n_assets
            else:
                # Fallback to mean-variance if insufficient data
                weights = self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
            
            return weights
            
        except Exception:
            # Fallback to mean-variance optimization
            return self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
    
    def _ridge_optimization(self, returns_data, expected_returns, cov_matrix, risk_tolerance):
        """Ridge regression for portfolio optimization with L2 regularization."""
        from sklearn.linear_model import Ridge
        
        try:
            n_assets = len(expected_returns)
            
            # Use Ridge regression for regularized optimization
            if len(returns_data) > 30:
                X = returns_data.iloc[:-1].values
                y = returns_data.mean().values
                
                # Fit Ridge model
                ridge = Ridge(alpha=1.0, positive=True)
                ridge.fit(X.T, y)
                
                # Get regularized weights
                weights = ridge.coef_
                weights = np.abs(weights)  # Ensure positive
                weights = weights / weights.sum()  # Normalize
                
                return weights
            else:
                # Fallback for insufficient data
                return self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
                
        except Exception:
            # Fallback to mean-variance
            return self._mean_variance_optimization(expected_returns, cov_matrix, risk_tolerance)
 