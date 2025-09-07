import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple

class Comparator:
    """Compare results between classical and quantum optimization methods."""
    
    def __init__(self):
        self.metrics = {}
    
    def compare_results(self, classical_result: Dict, quantum_result: Dict) -> Dict:
        """
        Compare classical and quantum optimization results.
        
        Args:
            classical_result: Results from classical optimizer
            quantum_result: Results from quantum optimizer
            
        Returns:
            Dictionary containing comparison metrics and visualizations
        """
        comparison = {
            'performance_metrics': self._compare_performance(classical_result, quantum_result),
            'allocation_comparison': self._compare_allocations(classical_result, quantum_result),
            'convergence_comparison': self._compare_convergence(classical_result, quantum_result),
            'computational_efficiency': self._compare_efficiency(classical_result, quantum_result),
            'visualizations': self._create_comparison_plots(classical_result, quantum_result)
        }
        
        return comparison
    
    def _compare_performance(self, classical: Dict, quantum: Dict) -> Dict:
        """Compare portfolio performance metrics."""
        metrics = {
            'returns': {
                'classical': classical.get('expected_return', 0),
                'quantum': quantum.get('expected_return', 0),
                'difference': quantum.get('expected_return', 0) - classical.get('expected_return', 0)
            },
            'risk': {
                'classical': classical.get('risk', 0),
                'quantum': quantum.get('risk', 0),
                'difference': quantum.get('risk', 0) - classical.get('risk', 0)
            },
            'sharpe_ratio': {
                'classical': classical.get('sharpe_ratio', 0),
                'quantum': quantum.get('sharpe_ratio', 0),
                'difference': quantum.get('sharpe_ratio', 0) - classical.get('sharpe_ratio', 0)
            }
        }
        
        # Determine winner for each metric
        metrics['winner'] = {
            'returns': 'quantum' if metrics['returns']['difference'] > 0 else 'classical',
            'risk': 'classical' if metrics['risk']['difference'] > 0 else 'quantum',  # Lower risk is better
            'sharpe_ratio': 'quantum' if metrics['sharpe_ratio']['difference'] > 0 else 'classical'
        }
        
        return metrics
    
    def _compare_allocations(self, classical: Dict, quantum: Dict) -> Dict:
        """Compare portfolio allocations."""
        classical_weights = classical.get('weights', np.array([]))
        quantum_weights = quantum.get('weights', np.array([]))
        
        # Ensure same length
        max_len = max(len(classical_weights), len(quantum_weights))
        if len(classical_weights) < max_len:
            classical_weights = np.pad(classical_weights, (0, max_len - len(classical_weights)))
        if len(quantum_weights) < max_len:
            quantum_weights = np.pad(quantum_weights, (0, max_len - len(quantum_weights)))
        
        # Calculate allocation metrics
        allocation_overlap = np.sum(np.minimum(classical_weights, quantum_weights))
        weight_correlation = np.corrcoef(classical_weights, quantum_weights)[0, 1] if len(classical_weights) > 1 else 0
        
        # L2 distance between allocations
        allocation_distance = np.linalg.norm(classical_weights - quantum_weights)
        
        # Number of assets selected
        classical_assets = np.sum(classical_weights > 0.01)  # Threshold for "selected"
        quantum_assets = np.sum(quantum_weights > 0.01)
        
        return {
            'overlap': allocation_overlap,
            'correlation': weight_correlation,
            'distance': allocation_distance,
            'assets_selected': {
                'classical': classical_assets,
                'quantum': quantum_assets,
                'difference': quantum_assets - classical_assets
            },
            'weights': {
                'classical': classical_weights,
                'quantum': quantum_weights
            }
        }
    
    def _compare_convergence(self, classical: Dict, quantum: Dict) -> Dict:
        """Compare convergence behavior."""
        classical_history = classical.get('convergence_history', [])
        quantum_history = quantum.get('convergence_history', [])
        
        convergence_metrics = {
            'classical_iterations': len(classical_history),
            'quantum_iterations': len(quantum_history),
            'classical_final_value': classical_history[-1] if classical_history else 0,
            'quantum_final_value': quantum_history[-1] if quantum_history else 0,
            'histories': {
                'classical': classical_history,
                'quantum': quantum_history
            }
        }
        
        # Calculate convergence rate (improvement per iteration)
        if len(classical_history) > 1:
            classical_rate = (classical_history[-1] - classical_history[0]) / len(classical_history)
        else:
            classical_rate = 0
            
        if len(quantum_history) > 1:
            quantum_rate = (quantum_history[-1] - quantum_history[0]) / len(quantum_history)
        else:
            quantum_rate = 0
        
        convergence_metrics['convergence_rates'] = {
            'classical': classical_rate,
            'quantum': quantum_rate
        }
        
        return convergence_metrics
    
    def _compare_efficiency(self, classical: Dict, quantum: Dict) -> Dict:
        """Compare computational efficiency."""
        classical_time = classical.get('computation_time', 0)
        quantum_time = quantum.get('computation_time', 0)
        
        efficiency = {
            'computation_time': {
                'classical': classical_time,
                'quantum': quantum_time,
                'speedup': classical_time / quantum_time if quantum_time > 0 else float('inf')
            },
            'method_details': {
                'classical': classical.get('method', 'unknown'),
                'quantum': {
                    'method': quantum.get('method', 'unknown'),
                    'circuit_depth': quantum.get('circuit_depth', 0),
                    'shots': quantum.get('shots', 0)
                }
            }
        }
        
        return efficiency
    
    def _create_comparison_plots(self, classical: Dict, quantum: Dict) -> Dict:
        """Create comparison visualizations."""
        plots = {}
        
        # 1. Performance comparison radar chart
        plots['performance_radar'] = self._create_performance_radar(classical, quantum)
        
        # 2. Allocation comparison bar chart
        plots['allocation_bars'] = self._create_allocation_comparison(classical, quantum)
        
        # 3. Convergence comparison line chart
        plots['convergence_lines'] = self._create_convergence_plot(classical, quantum)
        
        # 4. Risk-return scatter plot
        plots['risk_return_scatter'] = self._create_risk_return_plot(classical, quantum)
        
        return plots
    
    def _create_performance_radar(self, classical: Dict, quantum: Dict) -> go.Figure:
        """Create radar chart comparing performance metrics."""
        categories = ['Expected Return', 'Sharpe Ratio', 'Diversification', 'Stability']
        
        # Normalize metrics to 0-1 scale for radar chart
        classical_values = [
            min(classical.get('expected_return', 0) * 5, 1),  # Scale return
            min(classical.get('sharpe_ratio', 0) / 2, 1),     # Scale Sharpe
            min(np.sum(classical.get('weights', []) > 0.01) / 10, 1),  # Diversification
            min(1 - classical.get('risk', 0.2) / 0.5, 1)     # Stability (inverse of risk)
        ]
        
        quantum_values = [
            min(quantum.get('expected_return', 0) * 5, 1),
            min(quantum.get('sharpe_ratio', 0) / 2, 1),
            min(np.sum(quantum.get('weights', []) > 0.01) / 10, 1),
            min(1 - quantum.get('risk', 0.2) / 0.5, 1)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=classical_values + [classical_values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name='Classical',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=quantum_values + [quantum_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Quantum',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Performance Comparison"
        )
        
        return fig
    
    def _create_allocation_comparison(self, classical: Dict, quantum: Dict) -> go.Figure:
        """Create bar chart comparing allocations."""
        classical_weights = classical.get('weights', [])
        quantum_weights = quantum.get('weights', [])
        tickers = classical.get('tickers', [f'Asset_{i}' for i in range(len(classical_weights))])
        
        # Ensure same length
        max_len = max(len(classical_weights), len(quantum_weights), len(tickers))
        if len(classical_weights) < max_len:
            classical_weights = np.pad(classical_weights, (0, max_len - len(classical_weights)))
        if len(quantum_weights) < max_len:
            quantum_weights = np.pad(quantum_weights, (0, max_len - len(quantum_weights)))
        if len(tickers) < max_len:
            tickers.extend([f'Asset_{i}' for i in range(len(tickers), max_len)])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Classical',
            x=tickers,
            y=classical_weights,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Quantum',
            x=tickers,
            y=quantum_weights,
            marker_color='red'
        ))
        
        fig.update_layout(
            title='Portfolio Allocation Comparison',
            xaxis_title='Assets',
            yaxis_title='Weight',
            barmode='group'
        )
        
        return fig
    
    def _create_convergence_plot(self, classical: Dict, quantum: Dict) -> go.Figure:
        """Create convergence comparison plot."""
        fig = go.Figure()
        
        classical_history = classical.get('convergence_history', [])
        quantum_history = quantum.get('convergence_history', [])
        
        if classical_history:
            fig.add_trace(go.Scatter(
                x=list(range(len(classical_history))),
                y=classical_history,
                mode='lines+markers',
                name='Classical',
                line=dict(color='blue')
            ))
        
        if quantum_history:
            fig.add_trace(go.Scatter(
                x=list(range(len(quantum_history))),
                y=quantum_history,
                mode='lines+markers',
                name='Quantum',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title='Convergence Comparison',
            xaxis_title='Iteration',
            yaxis_title='Objective Value',
            showlegend=True
        )
        
        return fig
    
    def _create_risk_return_plot(self, classical: Dict, quantum: Dict) -> go.Figure:
        """Create risk-return scatter plot."""
        fig = go.Figure()
        
        # Classical point
        fig.add_trace(go.Scatter(
            x=[classical.get('risk', 0)],
            y=[classical.get('expected_return', 0)],
            mode='markers',
            name='Classical',
            marker=dict(size=15, color='blue'),
            text=['Classical Method'],
            textposition="top center"
        ))
        
        # Quantum point
        fig.add_trace(go.Scatter(
            x=[quantum.get('risk', 0)],
            y=[quantum.get('expected_return', 0)],
            mode='markers',
            name='Quantum',
            marker=dict(size=15, color='red'),
            text=['Quantum Method'],
            textposition="top center"
        ))
        
        fig.update_layout(
            title='Risk vs Return Comparison',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            showlegend=True
        )
        
        return fig