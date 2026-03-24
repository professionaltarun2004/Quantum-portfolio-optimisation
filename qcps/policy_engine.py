from typing import Dict, Any

class PolicyEngine:
    """
    QRL-Lite Policy Engine.
    Maps market regimes to dynamic parameters and supports adaptive reinforcement
    episodic updates based on out-of-sample objective feedback constraints.
    """
    
    def __init__(self, learning_rate: float = 0.05):
        self.learning_rate = learning_rate
        # Mathematical scalars bounding topological QUBO scaling operations.
        self.policy_map = {
            "BULL_STABLE": {"risk_aversion_q": 1.0, "constraint_scaling": 1.0, "diversification_weight": 1.0},
            "BULL_VOLATILE": {"risk_aversion_q": 1.5, "constraint_scaling": 0.8, "diversification_weight": 1.5},
            "BEAR_STABLE": {"risk_aversion_q": 2.0, "constraint_scaling": 0.7, "diversification_weight": 2.0},
            "BEAR_VOLATILE": {"risk_aversion_q": 2.5, "constraint_scaling": 0.5, "diversification_weight": 2.5},
            "BEAR_CRISIS": {"risk_aversion_q": 3.0, "constraint_scaling": 0.3, "diversification_weight": 3.0},
            "INSUFFICIENT_DATA": {"risk_aversion_q": 1.5, "constraint_scaling": 1.0, "diversification_weight": 1.0}
        }
        
    def get_action(self, market_state: Dict[str, Any]) -> Dict[str, float]:
        """ Maps descriptive state spaces onto the functional numerical decision planes. """
        regime = market_state.get('regime', 'INSUFFICIENT_DATA')
        action = self.policy_map.get(regime, self.policy_map["INSUFFICIENT_DATA"]).copy()
        
        metrics = market_state.get('metrics', {})
        if metrics.get('volatility', 0.15) > 0.30:
            action["risk_aversion_q"] *= 1.2  # Acute unconditional risk scaling override jump
            
        return action
        
    def update_policy(self, regime: str, realized_sharpe: float):
        """
        Lightweight episodic continuous learning path.
        If a recognized state's historical Q-scaling decision resulted in outsized 
        underperformance vs structural targets, the parameters vectorially slide 
        incrementally toward heavier restrictive safety.
        """
        if regime not in self.policy_map or regime == "INSUFFICIENT_DATA":
            return
            
        # Hard structural targeted heuristic: Sharpe Ratio optimization 1.0 goal
        target_sharpe = 1.0
        error = target_sharpe - realized_sharpe
        
        # Unidirectional penalty learning constraint.
        # It adapts by tightening barriers (shrinking risk pools) when failing.
        if error > 0:
            adjustment = 1.0 + (self.learning_rate * error)
            
            # Constrained bounding update to empirically prevent mathematically collapsing QUBO states
            current_q = self.policy_map[regime]["risk_aversion_q"]
            self.policy_map[regime]["risk_aversion_q"] = min(current_q * adjustment, 5.0)
            
            current_div = self.policy_map[regime]["diversification_weight"]
            self.policy_map[regime]["diversification_weight"] = min(current_div * adjustment, 5.0)
