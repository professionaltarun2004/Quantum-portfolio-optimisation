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
        Encode portfolio optimization as QUBO matrix with enhanced formulation.
        
        Args:
            returns_data: Historical returns data
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            QUBO matrix Q where objective is x^T Q x
        """
        n_assets = len(returns_data.columns)
        
        # Calculate enhanced statistics
        expected_returns = returns_data.mean().values
        cov_matrix = returns_data.cov().values
        
        # Normalize returns to prevent numerical issues
        return_scale = np.std(expected_returns) if np.std(expected_returns) > 1e-8 else 1.0
        expected_returns = expected_returns / return_scale
        
        # Scale covariance matrix
        risk_scale = np.mean(np.diag(cov_matrix)) if np.mean(np.diag(cov_matrix)) > 1e-8 else 1.0
        cov_matrix = cov_matrix / risk_scale
        
        Q = np.zeros((n_assets, n_assets))
        
        # Enhanced objective formulation
        risk_aversion = 1.0 / max(risk_tolerance, 0.1)
        
        # Diagonal terms: combine return maximization and individual risk
        for i in range(n_assets):
            # Return component (negative because we maximize)
            return_component = -expected_returns[i] * 2.0
            
            # Individual risk component (positive because we minimize)
            risk_component = cov_matrix[i, i] * risk_aversion
            
            Q[i, i] = return_component + risk_component
        
        # Off-diagonal terms: pairwise risk interactions
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Covariance penalty (diversification benefit)
                covar_penalty = cov_matrix[i, j] * risk_aversion * 0.5
                
                # Add correlation-based interaction
                corr_ij = cov_matrix[i, j] / (np.sqrt(cov_matrix[i, i] * cov_matrix[j, j]) + 1e-8)
                interaction_bonus = -abs(corr_ij) * 0.1  # Reward diversification
                
                Q[i, j] = covar_penalty + interaction_bonus
                Q[j, i] = Q[i, j]  # Symmetric
        
        # Enhanced constraint handling
        target_assets = max(3, min(7, n_assets // 2))  # Dynamic target based on problem size
        constraint_penalty = self.penalty_weight * 2.0
        
        # Soft constraint: encourage optimal number of assets
        # Penalty term: (sum(x_i) - target_assets)^2
        for i in range(n_assets):
            Q[i, i] += constraint_penalty * (1 - 2 * target_assets / n_assets)
        
        # Cross terms for constraint
        constraint_cross = constraint_penalty * 2.0 / (n_assets * n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                Q[i, j] += constraint_cross
                Q[j, i] = Q[i, j]
        
        # Add sector diversification bonus (simulate sector effects)
        sector_bonus = 0.05
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Simulate sector diversity (assets far apart in index are different sectors)
                sector_distance = abs(i - j) / n_assets
                if sector_distance > 0.3:  # Different "sectors"
                    Q[i, j] -= sector_bonus  # Bonus for cross-sector diversification
                    Q[j, i] = Q[i, j]
        
        # Ensure numerical stability
        Q = np.nan_to_num(Q, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Scale the entire matrix for better optimization
        matrix_scale = np.max(np.abs(Q)) if np.max(np.abs(Q)) > 1e-8 else 1.0
        Q = Q / matrix_scale * 10.0  # Scale to reasonable range
        
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