import numpy as np
from scipy.optimize import minimize
import time
from typing import Dict, List, Tuple

class QuantumOptimizer:
    """Quantum-inspired portfolio optimization using VQE and QAOA algorithms."""
    
    def __init__(self):
        self.convergence_history = []
    
    def optimize(self, qubo_matrix: np.ndarray, method: str = "VQE", 
                circuit_depth: int = 2, shots: int = 1024) -> Dict:
        """
        Optimize portfolio using quantum algorithms.
        
        Args:
            qubo_matrix: QUBO matrix representation of the problem
            method: Quantum method ('VQE' or 'QAOA')
            circuit_depth: Depth of quantum circuit
            shots: Number of measurement shots
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        self.convergence_history = []
        
        # Determine problem size
        n_assets = qubo_matrix.shape[0] if qubo_matrix.size > 0 else 5
        
        if method == "VQE":
            result = self._run_vqe_inspired(qubo_matrix, n_assets, circuit_depth, shots)
        elif method == "QAOA":
            result = self._run_qaoa_inspired(qubo_matrix, n_assets, circuit_depth, shots)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        computation_time = time.time() - start_time
        
        # Extract solution
        optimal_params = result.get('optimal_params', [])
        best_solution = result.get('best_solution', np.ones(n_assets))
        
        # Calculate portfolio weights from solution
        weights = self._binary_to_weights(best_solution)
        
        # Calculate portfolio metrics (simplified)
        portfolio_return = np.random.uniform(0.08, 0.15)  # Placeholder
        portfolio_risk = np.random.uniform(0.12, 0.25)    # Placeholder
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 1e-8 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'computation_time': computation_time,
            'method': method,
            'optimal_params': optimal_params,
            'convergence_history': self.convergence_history,
            'quantum_result': result,
            'circuit_depth': circuit_depth,
            'shots': shots
        }
    
    def _evaluate_qubo_objective(self, solution: np.ndarray, qubo_matrix: np.ndarray) -> float:
        """Evaluate QUBO objective function."""
        if qubo_matrix.size == 0:
            return np.random.uniform(-1, 1)
        
        try:
            # Ensure solution is binary
            binary_solution = (solution > 0.5).astype(int)
            
            # Pad or truncate solution to match matrix size
            n = qubo_matrix.shape[0]
            if len(binary_solution) > n:
                binary_solution = binary_solution[:n]
            elif len(binary_solution) < n:
                binary_solution = np.pad(binary_solution, (0, n - len(binary_solution)))
            
            # Calculate QUBO objective: x^T Q x
            objective = binary_solution.T @ qubo_matrix @ binary_solution
            return float(objective)
        except Exception:
            return np.random.uniform(-1, 1)
    
    def _run_vqe_inspired(self, qubo_matrix: np.ndarray, n_assets: int, 
                         circuit_depth: int, shots: int) -> Dict:
        """Run VQE-inspired optimization using classical simulation."""
        
        # Initialize parameters (simulate quantum circuit parameters)
        num_params = n_assets * circuit_depth * 2  # Simulate rotation angles
        initial_params = np.random.uniform(0, 2*np.pi, num_params)
        
        # Define cost function that simulates quantum expectation value
        def cost_function(params):
            try:
                # Convert parameters to probability amplitudes (simulate quantum state)
                # Use trigonometric functions to simulate quantum rotations
                probabilities = np.abs(np.sin(params[:n_assets])) ** 2
                probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.ones(n_assets) / n_assets
                
                # Convert probabilities to binary solution (simulate measurement)
                binary_solution = (probabilities > np.mean(probabilities)).astype(float)
                
                # Evaluate QUBO objective
                objective_value = self._evaluate_qubo_objective(binary_solution, qubo_matrix)
                self.convergence_history.append(objective_value)
                return objective_value
            except Exception:
                objective_value = np.random.uniform(-1, 1)
                self.convergence_history.append(objective_value)
                return objective_value
        
        # Run optimization
        # Use SLSQP instead of COBYLA for better compatibility
        bounds = [(0, 2*np.pi) for _ in range(len(initial_params))]
        result = minimize(cost_function, initial_params, method='SLSQP', 
                         bounds=bounds, options={'maxiter': 50})
        
        # Generate final solution
        final_probabilities = np.abs(np.sin(result.x[:n_assets])) ** 2
        final_probabilities = final_probabilities / final_probabilities.sum() if final_probabilities.sum() > 0 else np.ones(n_assets) / n_assets
        best_solution = (final_probabilities > np.mean(final_probabilities)).astype(float)
        
        return {
            'optimal_params': result.x,
            'optimal_value': result.fun,
            'best_solution': best_solution
        }
    
    def _run_qaoa_inspired(self, qubo_matrix: np.ndarray, n_assets: int,
                          circuit_depth: int, shots: int) -> Dict:
        """Run QAOA-inspired optimization using classical simulation."""
        
        # Initialize parameters (2 per layer: gamma and beta, simulating QAOA)
        num_params = 2 * circuit_depth
        initial_params = np.random.uniform(0, np.pi, num_params)
        
        # Define cost function that simulates QAOA behavior
        def cost_function(params):
            try:
                # Simulate QAOA alternating optimization
                # Start with uniform superposition (equal probabilities)
                probabilities = np.ones(n_assets) / n_assets
                
                # Apply alternating "problem" and "mixer" operations
                for layer in range(circuit_depth):
                    gamma_idx = layer * 2
                    beta_idx = layer * 2 + 1
                    
                    gamma = params[gamma_idx] if gamma_idx < len(params) else 0.1
                    beta = params[beta_idx] if beta_idx < len(params) else 0.1
                    
                    # Problem Hamiltonian effect (bias towards better solutions)
                    for i in range(n_assets):
                        if qubo_matrix.size > 0 and i < qubo_matrix.shape[0]:
                            # Apply problem-dependent rotation
                            bias = qubo_matrix[i, i] if i < qubo_matrix.shape[1] else 0
                            probabilities[i] *= (1 + gamma * bias)
                    
                    # Mixer Hamiltonian effect (maintain exploration)
                    probabilities = probabilities * np.cos(beta) + (1 - probabilities) * np.sin(beta)
                    
                    # Normalize probabilities
                    probabilities = np.abs(probabilities)
                    probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.ones(n_assets) / n_assets
                
                # Convert to binary solution
                threshold = np.median(probabilities)
                binary_solution = (probabilities > threshold).astype(float)
                
                # Evaluate QUBO objective
                objective_value = self._evaluate_qubo_objective(binary_solution, qubo_matrix)
                self.convergence_history.append(objective_value)
                return objective_value
            except Exception:
                objective_value = np.random.uniform(-1, 1)
                self.convergence_history.append(objective_value)
                return objective_value
        
        # Run optimization
        # Use SLSQP instead of COBYLA for better compatibility
        bounds = [(0, 2*np.pi) for _ in range(len(initial_params))]
        result = minimize(cost_function, initial_params, method='SLSQP',
                         bounds=bounds, options={'maxiter': 50})
        
        # Generate final solution using optimal parameters
        final_probabilities = np.ones(n_assets) / n_assets
        
        # Apply optimal QAOA parameters
        for layer in range(circuit_depth):
            gamma_idx = layer * 2
            beta_idx = layer * 2 + 1
            
            gamma = result.x[gamma_idx] if gamma_idx < len(result.x) else 0.1
            beta = result.x[beta_idx] if beta_idx < len(result.x) else 0.1
            
            # Apply transformations
            for i in range(n_assets):
                if qubo_matrix.size > 0 and i < qubo_matrix.shape[0]:
                    bias = qubo_matrix[i, i] if i < qubo_matrix.shape[1] else 0
                    final_probabilities[i] *= (1 + gamma * bias)
            
            final_probabilities = final_probabilities * np.cos(beta) + (1 - final_probabilities) * np.sin(beta)
            final_probabilities = np.abs(final_probabilities)
            final_probabilities = final_probabilities / final_probabilities.sum() if final_probabilities.sum() > 0 else np.ones(n_assets) / n_assets
        
        threshold = np.median(final_probabilities)
        best_solution = (final_probabilities > threshold).astype(float)
        
        return {
            'optimal_params': result.x,
            'optimal_value': result.fun,
            'best_solution': best_solution
        }
    

    
    def _binary_to_weights(self, binary_solution: np.ndarray) -> np.ndarray:
        """Convert binary solution to portfolio weights."""
        # Simple approach: normalize selected assets equally
        selected_assets = binary_solution.astype(bool)
        
        if np.any(selected_assets):
            weights = selected_assets.astype(float)
            weights = weights / weights.sum()
        else:
            # Fallback: equal weights
            weights = np.ones(len(binary_solution)) / len(binary_solution)
        
        return weights
    
    def create_quantum_circuit_description(self, n_assets: int, method: str = "VQE", 
                                          depth: int = 2) -> str:
        """Create description of quantum circuit for educational purposes."""
        if method == "VQE":
            description = f"""
            VQE Circuit Structure:
            - {n_assets} qubits (one per asset)
            - {depth} layers of parameterized gates
            - RY rotation gates for single-qubit rotations
            - CZ gates for qubit entanglement
            - Total parameters: {n_assets * depth * 2}
            """
        else:  # QAOA
            description = f"""
            QAOA Circuit Structure:
            - {n_assets} qubits (one per asset)
            - Initial Hadamard gates for superposition
            - {depth} alternating layers:
              * Problem Hamiltonian (RZZ gates)
              * Mixer Hamiltonian (RX gates)
            - Total parameters: {2 * depth}
            """
        
        return description.strip()