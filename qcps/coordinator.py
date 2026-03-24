import pandas as pd
import numpy as np
from typing import Dict, List, Any

from qcps.market_state import MarketStateEngine
from qcps.rl_policy_engine import RLPolicyEngine
from qcps.dynamic_qubo import DynamicQUBO
from qcps.noise_aware import NoiseAwareExecutor
from qcps.analyzer import DistributionAnalyzer

class QCPSCoordinator:
    """
    The orchestrator for the Quantum Cognitive Portfolio System.
    Executes the multi-period simulation loop temporally routing data logically through the 
    Market Engine -> Policy -> QUBO -> Quantum Executor -> Analyzer architecture,
    while realistically tracking structural transaction costs and enabling adaptive policy momentum.
    """
    
    def __init__(self, use_quantum: bool = True, use_rl: bool = True):
        self.market_engine = MarketStateEngine()
        self.use_rl = use_rl
        if use_rl:
            from qcps.rl_policy_engine import RLPolicyEngine
            self.policy_engine = RLPolicyEngine(learning_rate=0.01)
        else:
            from qcps.policy_engine import PolicyEngine
            self.policy_engine = PolicyEngine(learning_rate=0.05)
        self.dynamic_qubo = DynamicQUBO()
        self.executor = NoiseAwareExecutor(use_quantum=use_quantum)
        self.analyzer = DistributionAnalyzer()
        
    def run_simulation(self, 
                       historical_prices: pd.DataFrame, 
                       base_risk_tolerance: float = 1.0,
                       window_size: int = 126, # 6 months trailing lookback window
                       step_size: int = 21,    # ~1 month out-of-sample forward block evaluation
                       quantum_method: str = "QAOA"
                       ) -> Dict[str, Any]:
        """
        Runs an absolute out-of-sample sliding backtest implementing rigorous transaction tax matrices
        and episodic policy evaluation error updates.
        """
        if len(historical_prices) < window_size + step_size:
            return {'error': 'Not enough temporal data blocks to slice a comprehensive train-test simulation window.'}
            
        tickers = historical_prices.columns.tolist()
        simulation_results = []
        
        start_idx = window_size
        max_idx = len(historical_prices)
        
        while start_idx < max_idx:
            end_idx = min(start_idx + step_size, max_idx)
            
            train_data = historical_prices.iloc[:start_idx]
            test_data = historical_prices.iloc[start_idx:end_idx]
            train_returns = train_data.pct_change().dropna()
            
            # === PIPELINE PHASE 1: COGNITIVE STATE CLASSIFICATION ===
            market_state = self.market_engine.get_current_state(train_data)
            policy_action = self.policy_engine.get_action(market_state)
            
            # === PIPELINE PHASE 2: QUBO TOPOLOGICAL ENCODING ===
            Q_matrix = self.dynamic_qubo.encode_portfolio_problem(
                returns_data=train_returns,
                base_risk_tolerance=base_risk_tolerance,
                policy_action=policy_action
            )
            
            # === PIPELINE PHASE 3: OPEN-SYSTEM PROBABILISTIC EXECUTION ===
            ensemble = self.executor.optimize_ensemble(
                qubo_matrix=Q_matrix,
                method=quantum_method,
                returns_data=train_returns
            )
            distribution_result = self.analyzer.analyze(ensemble, tickers)
            
            # === PIPELINE PHASE 4: OOS TRUE EVALUATION & TRANSACTION TAXES ===
            test_returns = test_data.pct_change().dropna()
            weights = distribution_result.get('consensus_weights', np.ones(len(tickers))/len(tickers))
            
            # Realistic Transaction Penalty Mathematical Operations (15 bps institutional half-turn)
            transaction_cost_bps = 15.0 / 10000.0
            if start_idx == window_size:
                turnover = 1.0  # Entire internal balance analytically deployed initially
            else:
                prev_weights = np.array(simulation_results[-1]['weights'])
                turnover = np.sum(np.abs(weights - prev_weights)) / 2.0
                
            tx_costs = turnover * transaction_cost_bps
            
            if not test_returns.empty:
                period_returns = np.dot(test_returns.values, weights)
                period_total_return = np.prod(1 + period_returns) - 1
                
                # Directly apply unrecoverable market turnover drag constraint
                period_total_return -= tx_costs
                period_vol = np.std(period_returns) * np.sqrt(252) if len(period_returns) > 1 else 0.1
            else:
                period_total_return = 0
                period_vol = 0
                
            period_sharpe = (period_total_return - 0.02) / period_vol if period_vol > 0 else 0
            
            # Extract out-of-sample drawdown for deep reward mapping
            if not test_returns.empty:
                cum_oos = (1 + period_returns).cumprod()
                rol_max = np.maximum.accumulate(cum_oos)
                oos_drawdown = np.min(cum_oos / rol_max - 1.0) if len(cum_oos) > 0 else 0.0
            else:
                oos_drawdown = 0.0

            # === PIPELINE PHASE 5: QRL REINFORCEMENT LEARNING UPDATE ===
            if self.use_rl:
                self.policy_engine.store_experience(sharpe=period_sharpe, drawdown=oos_drawdown, tx_cost=tx_costs)
                self.policy_engine.update_policy()
            else:
                self.policy_engine.update_policy(market_state['regime'], period_sharpe)
            
            step_record = {
                'step_start': test_data.index[0],
                'step_end': test_data.index[-1],
                'market_regime': market_state['regime'],
                'market_metrics': market_state.get('metrics', {}),
                'policy_action': policy_action,
                'qubo_matrix': Q_matrix.tolist(),
                'ensemble_results': ensemble,
                'distribution_metrics': distribution_result.get('metrics', {}),
                'stability_score': distribution_result.get('stability_score', 0),
                'turnover': turnover,
                'tx_costs': tx_costs,
                'weights': weights.tolist(),
                'realized_return': period_total_return,
                'realized_volatility': period_vol,
                'realized_sharpe': period_sharpe
            }
            
            simulation_results.append(step_record)
            start_idx += step_size
            
        if not simulation_results:
             return {'error': 'Target boundary simulation fundamentally failed to generate outputs.'}
             
        # Compile global structural trajectory evaluations
        total_compound = np.prod([1 + r['realized_return'] for r in simulation_results]) - 1
        avg_vol = np.mean([r['realized_volatility'] for r in simulation_results])
        realized_sharpe = (total_compound - 0.02) / avg_vol if avg_vol > 0 else 0
        total_tx_costs = sum(r['tx_costs'] for r in simulation_results)
        
        return {
            'historical_trajectory': simulation_results,
            'summary': {
                'total_return': total_compound,
                'average_volatility': avg_vol,
                'realized_sharpe': realized_sharpe,
                'total_transaction_costs': total_tx_costs,
                'average_stability': float(np.mean([r['stability_score'] for r in simulation_results])),
                'steps_run': len(simulation_results)
            }
        }
