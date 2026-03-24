import numpy as np
import pandas as pd
from typing import Dict, Any

from backend.qubo_encoder import QUBOEncoder

class DynamicQUBO:
    """
    Dynamic QUBO Wrapper for the Quantum Cognitive Portfolio System.
    Wraps the standard QUBOEncoder and applies Policy Engine scaling factors
    without fundamentally rewriting the base logic.
    """
    
    def __init__(self):
        self.base_encoder = QUBOEncoder()

    def encode_portfolio_problem(self, 
                                 returns_data: pd.DataFrame, 
                                 base_risk_tolerance: float, 
                                 policy_action: Dict[str, float]) -> np.ndarray:
        """
        Dynamically adjusts risk tolerance, penalties, and interactions 
        based on the market regime policy before encoding.
        """
        # 1. Extract Policy Actions
        risk_aversion_q = policy_action.get('risk_aversion_q', 1.0)
        constraint_scale = policy_action.get('constraint_scaling', 1.0)
        div_weight = policy_action.get('diversification_weight', 1.0)
        
        # 2. Adjust standard parameters mathematically based on policy
        effective_risk_tolerance = base_risk_tolerance / max(risk_aversion_q, 0.1)
        
        # Dynamically scale the raw penalty weight of the encoder
        original_penalty = self.base_encoder.penalty_weight
        self.base_encoder.penalty_weight = original_penalty / max(constraint_scale, 0.1)
        
        # 3. Process Returns 
        cov_matrix = returns_data.cov()
        
        # Generate the baseline QUBO using the base system's proven logic
        Q = self.base_encoder.encode_portfolio_problem(returns_data, risk_tolerance=effective_risk_tolerance)
        
        # Restore penalty to baseline
        self.base_encoder.penalty_weight = original_penalty
        
        # 4. Post-process QUBO matrix for spatial 'diversification_weight' mapped from the QRL Policy
        n_assets = len(returns_data.columns)
        if Q.shape == (n_assets, n_assets):
            cov_vals = cov_matrix.values
            risk_scale = np.mean(np.diag(cov_vals)) if np.mean(np.diag(cov_vals)) > 1e-8 else 1.0
            cov_vals = cov_vals / risk_scale
            
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    corr_ij = cov_vals[i, j] / (np.sqrt(cov_vals[i, i] * cov_vals[j, j]) + 1e-8)
                    
                    # The base system added an extra bonus `-abs(corr_ij)*0.1`. 
                    # We inject our scaling multiplier onto that bonus.
                    extra_bonus = -abs(corr_ij) * 0.1 * (div_weight - 1.0)
                    
                    Q[i, j] += extra_bonus
                    Q[j, i] += extra_bonus
                    
        return Q
