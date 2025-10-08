import numpy as np
import pandas as pd
from typing import Dict, Tuple

class QUBOEncoder:
    """Encodes portfolio optimization problems as QUBO (Quadratic Unconstrained Binary Optimization)."""
    
    def __init__(self):
        self.penalty_weight = 10.0
        self.discretization_levels = 10
    
    def encode_portfolio_problem(self, returns_data: pd.DataFrame, risk_tolerance: float = 1.0) -> np.ndarray:
        """
        Encode portfolio optimization as QUBO matrix.
        
        Args:
            returns_data: Historical returns data
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            QUBO matrix Q where objective is x^T Q x
        """
        n_assets = len(returns_data.columns)
        expected_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        # For simplicity, use binary variables for asset selection
        # Each asset can be either included (1) or not (0)
        # We'll add penalty terms to enforce portfolio constraints
        
        Q = np.zeros((n_assets, n_assets))
        
        # Diagonal terms: -expected_return (we want to maximize returns)
        for i in range(n_assets):
            Q[i, i] = -expected_returns[i] / max(risk_tolerance, 1e-8)
        
        # Off-diagonal terms: covariance (we want to minimize risk)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                Q[i, j] = cov_matrix[i, j] / (2 * max(risk_tolerance, 1e-8))
                Q[j, i] = Q[i, j]  # Symmetric
        
        # Add penalty for having too few or too many assets
        target_assets = min(5, n_assets)
        penalty = self.penalty_weight
        
        # Penalty term: (sum(x_i) - target_assets)^2
        # Expanded: sum(x_i^2) + target_assets^2 - 2*target_assets*sum(x_i) + 2*sum_i<j(x_i*x_j)
        
        # x_i^2 = x_i for binary variables, so add to diagonal
        for i in range(n_assets):
            Q[i, i] += penalty * (1 - 2 * target_assets)
        
        # Cross terms: 2*x_i*x_j
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                Q[i, j] += penalty
                Q[j, i] = Q[i, j]
        
        return Q
    
    def encode_continuous_portfolio(self, returns_data: pd.DataFrame, risk_tolerance: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Encode portfolio with continuous weights using binary encoding.
        
        Each weight is represented by multiple binary variables for discretization.
        """
        n_assets = len(returns_data.columns)
        n_bits_per_weight = 4  # 16 levels per weight
        total_vars = n_assets * n_bits_per_weight
        
        expected_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        Q = np.zeros((total_vars, total_vars))
        
        # Map binary variables to weights
        # weight_i = sum_k(2^k * x_{i,k}) / (2^n_bits - 1)
        
        for asset_i in range(n_assets):
            for asset_j in range(n_assets):
                for bit_i in range(n_bits_per_weight):
                    for bit_j in range(n_bits_per_weight):
                        var_i = asset_i * n_bits_per_weight + bit_i
                        var_j = asset_j * n_bits_per_weight + bit_j
                        
                        weight_i_coeff = (2 ** bit_i) / (2 ** n_bits_per_weight - 1)
                        weight_j_coeff = (2 ** bit_j) / (2 ** n_bits_per_weight - 1)
                        
                        if asset_i == asset_j:
                            # Return term (diagonal)
                            if bit_i == bit_j:
                                Q[var_i, var_j] += -expected_returns[asset_i] * weight_i_coeff / max(risk_tolerance, 1e-8)
                            
                            # Risk term (diagonal of covariance)
                            Q[var_i, var_j] += cov_matrix[asset_i, asset_j] * weight_i_coeff * weight_j_coeff / risk_tolerance
                        else:
                            # Risk term (off-diagonal of covariance)
                            Q[var_i, var_j] += cov_matrix[asset_i, asset_j] * weight_i_coeff * weight_j_coeff / risk_tolerance
        
        # Add constraint: sum of weights = 1
        penalty = self.penalty_weight * 10
        
        # (sum_i sum_k weight_coeff_{i,k} * x_{i,k} - 1)^2
        for asset_i in range(n_assets):
            for bit_i in range(n_bits_per_weight):
                var_i = asset_i * n_bits_per_weight + bit_i
                weight_coeff_i = (2 ** bit_i) / (2 ** n_bits_per_weight - 1)
                
                # Linear term: -2 * weight_coeff_i
                Q[var_i, var_i] += penalty * (-2 * weight_coeff_i + weight_coeff_i ** 2)
                
                # Cross terms
                for asset_j in range(n_assets):
                    for bit_j in range(n_bits_per_weight):
                        if asset_i != asset_j or bit_i != bit_j:
                            var_j = asset_j * n_bits_per_weight + bit_j
                            weight_coeff_j = (2 ** bit_j) / (2 ** n_bits_per_weight - 1)
                            Q[var_i, var_j] += penalty * weight_coeff_i * weight_coeff_j
        
        # Constant term (penalty * 1^2) is ignored in QUBO
        
        encoding_info = {
            'n_assets': n_assets,
            'n_bits_per_weight': n_bits_per_weight,
            'total_vars': total_vars,
            'tickers': returns_data.columns.tolist()
        }
        
        return Q, encoding_info
    
    def decode_binary_solution(self, solution: np.ndarray, encoding_info: Dict) -> np.ndarray:
        """Decode binary solution back to portfolio weights."""
        n_assets = encoding_info['n_assets']
        n_bits_per_weight = encoding_info['n_bits_per_weight']
        
        weights = np.zeros(n_assets)
        
        for asset_i in range(n_assets):
            weight_value = 0
            for bit_j in range(n_bits_per_weight):
                var_idx = asset_i * n_bits_per_weight + bit_j
                if var_idx < len(solution):
                    weight_value += solution[var_idx] * (2 ** bit_j)
            
            weights[asset_i] = weight_value / (2 ** n_bits_per_weight - 1)
        
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def validate_qubo_matrix(self, Q: np.ndarray) -> bool:
        """Validate that QUBO matrix is properly formed."""
        # Check if matrix is square
        if Q.shape[0] != Q.shape[1]:
            return False
        
        # Check if matrix is symmetric (within tolerance)
        if not np.allclose(Q, Q.T, rtol=1e-10):
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
            return False
        
        return True