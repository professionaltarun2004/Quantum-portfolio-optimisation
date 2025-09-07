import numpy as np
from typing import Dict, List, Tuple

class ExplanationLayer:
    """Generate educational explanations and insights about optimization results."""
    
    def __init__(self):
        self.explanations = {}
    
    def generate_explanations(self, results: Dict) -> Dict:
        """
        Generate comprehensive explanations for optimization results.
        
        Args:
            results: Dictionary containing classical and/or quantum results
            
        Returns:
            Dictionary with explanations, insights, and educational content
        """
        explanations = {
            'method_explanations': self._explain_methods(results),
            'result_insights': self._generate_insights(results),
            'theoretical_background': self._get_theoretical_background(),
            'practical_implications': self._analyze_practical_implications(results),
            'educational_resources': self._get_educational_resources(),
            'algorithm_details': self._explain_algorithms(results)
        }
        
        return explanations
    
    def _explain_methods(self, results: Dict) -> Dict:
        """Explain the optimization methods used."""
        explanations = {}
        
        if 'classical' in results:
            classical_method = results['classical'].get('method', 'mean_variance')
            explanations['classical'] = {
                'title': 'Classical Optimization',
                'method': classical_method,
                'description': self._get_classical_description(classical_method),
                'advantages': [
                    'Well-established mathematical foundation',
                    'Fast computation for moderate-sized problems',
                    'Proven track record in financial industry',
                    'Interpretable results and parameters'
                ],
                'limitations': [
                    'Assumes normal distribution of returns',
                    'May struggle with complex constraints',
                    'Limited ability to handle discrete variables',
                    'Computational complexity grows with problem size'
                ]
            }
        
        if 'quantum' in results:
            quantum_method = results['quantum'].get('method', 'VQE')
            explanations['quantum'] = {
                'title': 'Quantum Optimization',
                'method': quantum_method,
                'description': self._get_quantum_description(quantum_method),
                'advantages': [
                    'Potential exponential speedup for certain problems',
                    'Natural handling of combinatorial optimization',
                    'Can explore solution space more efficiently',
                    'Promising for large-scale portfolio problems'
                ],
                'limitations': [
                    'Current quantum computers are noisy (NISQ era)',
                    'Limited number of qubits available',
                    'Requires problem reformulation as QUBO',
                    'Results may vary due to quantum noise'
                ]
            }
        
        return explanations
    
    def _get_classical_description(self, method: str) -> str:
        """Get description for classical optimization method."""
        descriptions = {
            'mean_variance': """
            Mean-Variance Optimization (Markowitz Model): This classical approach finds the optimal 
            portfolio by balancing expected returns against risk (variance). It solves a quadratic 
            programming problem to find weights that maximize the utility function: 
            U = μᵀw - (λ/2)wᵀΣw, where μ is expected returns, Σ is covariance matrix, 
            and λ is risk aversion parameter.
            """,
            'ml_enhanced': """
            ML-Enhanced Optimization: This approach combines traditional mean-variance optimization 
            with machine learning techniques. It uses Random Forest regression to predict future 
            returns based on technical indicators, then applies these enhanced predictions in 
            the optimization process.
            """
        }
        return descriptions.get(method, "Classical optimization method")
    
    def _get_quantum_description(self, method: str) -> str:
        """Get description for quantum optimization method."""
        descriptions = {
            'VQE': """
            Variational Quantum Eigensolver (VQE): A hybrid quantum-classical algorithm that 
            finds the minimum eigenvalue of a Hamiltonian. For portfolio optimization, the 
            problem is encoded as a QUBO and converted to an Ising Hamiltonian. VQE uses 
            a parameterized quantum circuit (ansatz) and classical optimization to minimize 
            the expectation value ⟨ψ(θ)|H|ψ(θ)⟩.
            """,
            'QAOA': """
            Quantum Approximate Optimization Algorithm (QAOA): Specifically designed for 
            combinatorial optimization problems. QAOA alternates between problem Hamiltonian 
            (encoding the objective) and mixer Hamiltonian (enabling transitions between states). 
            The algorithm uses p layers of these alternating unitaries: |ψ⟩ = U(B,β)U(C,γ)|+⟩.
            """
        }
        return descriptions.get(method, "Quantum optimization method")
    
    def _generate_insights(self, results: Dict) -> Dict:
        """Generate insights about the optimization results."""
        insights = {}
        
        if 'classical' in results and 'quantum' in results:
            classical = results['classical']
            quantum = results['quantum']
            
            # Performance insights
            return_diff = quantum.get('expected_return', 0) - classical.get('expected_return', 0)
            risk_diff = quantum.get('risk', 0) - classical.get('risk', 0)
            
            insights['performance'] = {
                'return_comparison': self._interpret_return_difference(return_diff),
                'risk_comparison': self._interpret_risk_difference(risk_diff),
                'overall_assessment': self._assess_overall_performance(classical, quantum)
            }
            
            # Allocation insights
            classical_weights = classical.get('weights', [])
            quantum_weights = quantum.get('weights', [])
            
            if len(classical_weights) > 0 and len(quantum_weights) > 0:
                insights['allocation'] = {
                    'diversification': self._compare_diversification(classical_weights, quantum_weights),
                    'concentration': self._analyze_concentration(classical_weights, quantum_weights),
                    'similarity': self._measure_allocation_similarity(classical_weights, quantum_weights)
                }
            
            # Computational insights
            insights['computation'] = {
                'efficiency': self._compare_computational_efficiency(classical, quantum),
                'scalability': self._discuss_scalability(classical, quantum)
            }
        
        return insights
    
    def _interpret_return_difference(self, diff: float) -> str:
        """Interpret the difference in expected returns."""
        if abs(diff) < 0.01:
            return "Both methods achieved similar expected returns, suggesting comparable optimization quality."
        elif diff > 0:
            return f"Quantum optimization achieved {diff:.2%} higher expected return, potentially due to better exploration of the solution space."
        else:
            return f"Classical optimization achieved {abs(diff):.2%} higher expected return, likely due to more mature optimization techniques."
    
    def _interpret_risk_difference(self, diff: float) -> str:
        """Interpret the difference in portfolio risk."""
        if abs(diff) < 0.01:
            return "Both methods resulted in similar risk levels."
        elif diff > 0:
            return f"Quantum method resulted in {diff:.2%} higher risk, which may be due to quantum noise or different optimization objectives."
        else:
            return f"Quantum method achieved {abs(diff):.2%} lower risk, suggesting effective risk management through quantum optimization."
    
    def _assess_overall_performance(self, classical: Dict, quantum: Dict) -> str:
        """Provide overall performance assessment."""
        classical_sharpe = classical.get('sharpe_ratio', 0)
        quantum_sharpe = quantum.get('sharpe_ratio', 0)
        
        if abs(classical_sharpe - quantum_sharpe) < 0.1:
            return "Both methods show comparable risk-adjusted performance (Sharpe ratio)."
        elif quantum_sharpe > classical_sharpe:
            return f"Quantum optimization shows superior risk-adjusted performance with {quantum_sharpe:.2f} vs {classical_sharpe:.2f} Sharpe ratio."
        else:
            return f"Classical optimization demonstrates better risk-adjusted performance with {classical_sharpe:.2f} vs {quantum_sharpe:.2f} Sharpe ratio."
    
    def _compare_diversification(self, classical_weights: np.ndarray, quantum_weights: np.ndarray) -> str:
        """Compare diversification between methods."""
        # Calculate Herfindahl index (concentration measure)
        classical_hhi = np.sum(classical_weights ** 2)
        quantum_hhi = np.sum(quantum_weights ** 2)
        
        if abs(classical_hhi - quantum_hhi) < 0.05:
            return "Both methods achieved similar diversification levels."
        elif quantum_hhi < classical_hhi:
            return "Quantum optimization resulted in better diversification (lower concentration)."
        else:
            return "Classical optimization achieved better diversification."
    
    def _analyze_concentration(self, classical_weights: np.ndarray, quantum_weights: np.ndarray) -> str:
        """Analyze portfolio concentration."""
        classical_top3 = np.sum(np.sort(classical_weights)[-3:])
        quantum_top3 = np.sum(np.sort(quantum_weights)[-3:])
        
        return f"Top 3 holdings represent {classical_top3:.1%} (classical) vs {quantum_top3:.1%} (quantum) of the portfolio."
    
    def _measure_allocation_similarity(self, classical_weights: np.ndarray, quantum_weights: np.ndarray) -> str:
        """Measure similarity between allocations."""
        # Ensure same length
        min_len = min(len(classical_weights), len(quantum_weights))
        if min_len > 0:
            correlation = np.corrcoef(classical_weights[:min_len], quantum_weights[:min_len])[0, 1]
            
            if correlation > 0.8:
                return f"High similarity between allocations (correlation: {correlation:.2f})"
            elif correlation > 0.5:
                return f"Moderate similarity between allocations (correlation: {correlation:.2f})"
            else:
                return f"Low similarity between allocations (correlation: {correlation:.2f})"
        return "Cannot compare allocations due to different dimensions."
    
    def _compare_computational_efficiency(self, classical: Dict, quantum: Dict) -> str:
        """Compare computational efficiency."""
        classical_time = classical.get('computation_time', 0)
        quantum_time = quantum.get('computation_time', 0)
        
        if quantum_time > 0:
            speedup = classical_time / quantum_time
            if speedup > 1.5:
                return f"Quantum method was {speedup:.1f}x faster than classical."
            elif speedup < 0.67:
                return f"Classical method was {1/speedup:.1f}x faster than quantum."
            else:
                return "Both methods had similar computation times."
        return "Cannot compare computation times."
    
    def _discuss_scalability(self, classical: Dict, quantum: Dict) -> str:
        """Discuss scalability implications."""
        return """
        Scalability Considerations:
        - Classical methods scale polynomially with problem size but may become slow for large portfolios
        - Quantum methods could offer exponential advantages for large-scale problems but are limited by current hardware
        - Current quantum computers (NISQ era) are suitable for small to medium portfolio sizes
        - Future fault-tolerant quantum computers may enable optimization of very large portfolios
        """
    
    def _get_theoretical_background(self) -> Dict:
        """Provide theoretical background information."""
        return {
            'portfolio_theory': {
                'title': 'Modern Portfolio Theory',
                'content': """
                Developed by Harry Markowitz in 1952, Modern Portfolio Theory provides a mathematical 
                framework for constructing portfolios that maximize expected return for a given level 
                of risk. The key insight is that portfolio risk depends not only on individual asset 
                risks but also on correlations between assets.
                
                Key Concepts:
                - Efficient Frontier: The set of optimal portfolios offering the highest expected return for each risk level
                - Diversification: Reducing risk through combining uncorrelated assets
                - Risk-Return Tradeoff: The fundamental principle that higher returns require accepting higher risk
                """
            },
            'quantum_computing': {
                'title': 'Quantum Computing for Optimization',
                'content': """
                Quantum computers leverage quantum mechanical phenomena like superposition and entanglement 
                to process information in fundamentally different ways than classical computers. For optimization:
                
                Key Advantages:
                - Superposition allows exploring multiple solutions simultaneously
                - Quantum interference can amplify correct solutions and cancel incorrect ones
                - Entanglement enables complex correlations between variables
                
                Current Limitations:
                - Quantum decoherence limits computation time
                - Gate errors introduce noise in calculations
                - Limited number of qubits restricts problem size
                """
            }
        }
    
    def _analyze_practical_implications(self, results: Dict) -> Dict:
        """Analyze practical implications of the results."""
        return {
            'investment_perspective': """
            From an investment management perspective, these results demonstrate:
            1. Both classical and quantum methods can produce viable portfolio allocations
            2. The choice of method may depend on specific constraints and objectives
            3. Quantum methods may offer advantages for complex, multi-objective optimization
            4. Classical methods remain reliable and well-understood for most applications
            """,
            'technology_readiness': """
            Current Technology Status:
            - Classical optimization: Production-ready, widely used in industry
            - Quantum optimization: Research/experimental stage, showing promise
            - Hybrid approaches: Emerging as practical near-term solutions
            - Timeline: Quantum advantage may emerge in 5-10 years for specific problems
            """,
            'recommendations': """
            Practical Recommendations:
            1. Use classical methods for current production systems
            2. Experiment with quantum methods for research and future preparation
            3. Consider hybrid classical-quantum approaches for complex problems
            4. Monitor quantum hardware developments for future opportunities
            """
        }
    
    def _get_educational_resources(self) -> Dict:
        """Provide educational resources and references."""
        return {
            'books': [
                "Portfolio Selection by Harry Markowitz",
                "Quantum Computing: An Applied Approach by Hidary",
                "Quantum Computation and Quantum Information by Nielsen & Chuang"
            ],
            'papers': [
                "Portfolio optimization with quantum computers (Mugel et al., 2020)",
                "Quantum algorithms for portfolio optimization (Orus et al., 2019)",
                "QAOA for Max-Cut (Farhi et al., 2014)"
            ],
            'online_courses': [
                "Qiskit Textbook - Quantum Algorithms for Applications",
                "IBM Quantum Experience",
                "Microsoft Quantum Development Kit"
            ],
            'videos': [
                "Introduction to QAOA - IBM Quantum",
                "VQE Tutorial - Qiskit",
                "Portfolio Theory Explained - Khan Academy"
            ]
        }
    
    def _explain_algorithms(self, results: Dict) -> Dict:
        """Provide detailed algorithm explanations."""
        algorithms = {}
        
        if 'classical' in results:
            algorithms['classical'] = {
                'mathematical_formulation': """
                Minimize: (1/2) * w^T * Σ * w - λ * μ^T * w
                Subject to: Σw_i = 1, w_i ≥ 0
                
                Where:
                - w: portfolio weights vector
                - Σ: covariance matrix of returns
                - μ: expected returns vector
                - λ: risk tolerance parameter
                """,
                'solution_method': "Quadratic Programming using interior-point methods",
                'complexity': "O(n³) where n is the number of assets"
            }
        
        if 'quantum' in results:
            quantum_method = results['quantum'].get('method', 'VQE')
            algorithms['quantum'] = {
                'mathematical_formulation': """
                QUBO Formulation: min x^T * Q * x
                Where Q encodes the portfolio optimization objective and constraints
                
                Quantum State: |ψ(θ)⟩ = U(θ)|0⟩
                Objective: min ⟨ψ(θ)|H|ψ(θ)⟩
                """,
                'solution_method': f"{quantum_method} with classical parameter optimization",
                'complexity': "Depends on circuit depth and number of parameters"
            }
        
        return algorithms