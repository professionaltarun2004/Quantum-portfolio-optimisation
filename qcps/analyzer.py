import numpy as np
from typing import List, Dict, Any

class DistributionAnalyzer:
    """
    Analyzes an ensemble of quantum optimizer results to construct 
    a probabilistic portfolio distribution and extract stability metrics.
    """
    
    def analyze(self, ensemble_results: List[Dict[str, Any]], tickers: List[str]) -> Dict[str, Any]:
        """
        Computes the consensus mean portfolio, the variance across the ensemble,
        and stability scores representing quantum confidence.
        """
        n_assets = len(tickers)
        if not ensemble_results:
            return {}
            
        # Extract all weight arrays and their ensemble probability weights
        weight_matrix = []
        prob_weights = []
        
        for res in ensemble_results:
            w = res.get('weights', np.zeros(n_assets))
            # Pad or truncate if dimensions unexpectedly mismatch
            if len(w) > n_assets: w = w[:n_assets]
            elif len(w) < n_assets: w = np.pad(w, (0, n_assets - len(w)))
            
            weight_matrix.append(w)
            prob_weights.append(res.get('_ensemble_weight', 1.0))
            
        weight_matrix = np.array(weight_matrix) # Shape: (N_runs, N_assets)
        prob_weights = np.array(prob_weights)
        prob_weights = prob_weights / np.sum(prob_weights) # Normalize
        
        # 1. Consensus Mean Portfolio (Weighted Average)
        consensus_weights = np.average(weight_matrix, axis=0, weights=prob_weights)
        
        # Normalize strictly to 1.0
        if np.sum(consensus_weights) > 0:
            consensus_weights = consensus_weights / np.sum(consensus_weights)
            
        # 2. Portfolio Variance (Uncertainty per asset)
        # Higher variance means the quantum solvers disagreed with each other heavily across noise levels
        asset_variance = np.average((weight_matrix - consensus_weights)**2, axis=0, weights=prob_weights)
        
        # 3. Overall Stability Score
        # Inverse of total variance. Higher is more stable.
        total_variance = np.sum(asset_variance)
        stability_score = 1.0 / (1.0 + total_variance * 10.0) # Scaled structural metric
        
        # 4. Asset Selection Frequency
        # Count how many times an asset had >1% weight across the trials
        selection_frequency = np.average((weight_matrix > 0.01).astype(float), axis=0, weights=prob_weights)
        
        # Calculate expected performance of the consensus portfolio
        mean_return = np.average([r.get('expected_return', 0) for r in ensemble_results], weights=prob_weights)
        mean_risk = np.average([r.get('risk', 0) for r in ensemble_results], weights=prob_weights)
        mean_sharpe = (mean_return - 0.02) / mean_risk if mean_risk > 1e-8 else 0
        
        return {
            'consensus_weights': consensus_weights,
            'asset_variance': asset_variance,
            'stability_score': stability_score,
            'selection_frequency': selection_frequency,
            'metrics': {
                'expected_return': mean_return,
                'risk': mean_risk,
                'sharpe_ratio': mean_sharpe
            },
            'tickers': tickers,
            'raw_ensemble': ensemble_results
        }
