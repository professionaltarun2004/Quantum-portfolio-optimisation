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
        Compare classical and quantum optimization results with user-friendly explanations.
        
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
            'visualizations': self._create_comparison_plots(classical_result, quantum_result),
            'user_friendly_summary': self._create_user_friendly_summary(classical_result, quantum_result)
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
   
    def _create_user_friendly_summary(self, classical: Dict, quantum: Dict) -> Dict:
        """Create user-friendly explanations of the comparison results."""
        
        # Calculate key metrics
        classical_return = classical.get('expected_return', 0) * 100  # Convert to percentage
        quantum_return = quantum.get('expected_return', 0) * 100
        classical_risk = classical.get('risk', 0) * 100
        quantum_risk = quantum.get('risk', 0) * 100
        classical_sharpe = classical.get('sharpe_ratio', 0)
        quantum_sharpe = quantum.get('sharpe_ratio', 0)
        
        # Determine which method performed better
        return_winner = "Quantum" if quantum_return > classical_return else "Classical"
        risk_winner = "Quantum" if quantum_risk < classical_risk else "Classical"  # Lower risk is better
        sharpe_winner = "Quantum" if quantum_sharpe > classical_sharpe else "Classical"
        
        # Create easy-to-understand explanations
        summary = {
            'headline': self._get_headline_summary(classical, quantum),
            'returns_explanation': {
                'winner': return_winner,
                'classical_return': f"{classical_return:.2f}%",
                'quantum_return': f"{quantum_return:.2f}%",
                'difference': f"{abs(quantum_return - classical_return):.2f}%",
                'explanation': self._explain_returns_difference(classical_return, quantum_return)
            },
            'risk_explanation': {
                'winner': risk_winner,
                'classical_risk': f"{classical_risk:.2f}%",
                'quantum_risk': f"{quantum_risk:.2f}%",
                'difference': f"{abs(quantum_risk - classical_risk):.2f}%",
                'explanation': self._explain_risk_difference(classical_risk, quantum_risk)
            },
            'efficiency_explanation': {
                'classical_sharpe': f"{classical_sharpe:.3f}",
                'quantum_sharpe': f"{quantum_sharpe:.3f}",
                'winner': sharpe_winner,
                'explanation': self._explain_sharpe_ratio(classical_sharpe, quantum_sharpe)
            },
            'diversification_analysis': self._analyze_diversification(classical, quantum),
            'practical_recommendation': self._get_practical_recommendation(classical, quantum),
            'computation_comparison': self._explain_computation_time(classical, quantum)
        }
        
        return summary
    
    def _get_headline_summary(self, classical: Dict, quantum: Dict) -> str:
        """Generate a headline summary of the comparison."""
        classical_sharpe = classical.get('sharpe_ratio', 0)
        quantum_sharpe = quantum.get('sharpe_ratio', 0)
        
        if abs(quantum_sharpe - classical_sharpe) < 0.05:
            return "📊 Both methods show similar performance - choose based on your preference!"
        elif quantum_sharpe > classical_sharpe:
            improvement = ((quantum_sharpe - classical_sharpe) / classical_sharpe * 100) if classical_sharpe > 0 else 0
            return f"🚀 Quantum method outperformed classical by {improvement:.1f}% in risk-adjusted returns!"
        else:
            improvement = ((classical_sharpe - quantum_sharpe) / quantum_sharpe * 100) if quantum_sharpe > 0 else 0
            return f"📈 Classical method outperformed quantum by {improvement:.1f}% in risk-adjusted returns!"
    
    def _explain_returns_difference(self, classical_return: float, quantum_return: float) -> str:
        """Explain the difference in expected returns."""
        diff = quantum_return - classical_return
        
        if abs(diff) < 0.5:
            return "Both methods predict very similar returns for your portfolio."
        elif diff > 0:
            return f"The quantum method suggests your portfolio could earn {diff:.2f}% more annually. This could mean an extra ${diff*10:.0f} per $1,000 invested each year."
        else:
            return f"The classical method suggests {abs(diff):.2f}% higher returns. This could mean an extra ${abs(diff)*10:.0f} per $1,000 invested each year."
    
    def _explain_risk_difference(self, classical_risk: float, quantum_risk: float) -> str:
        """Explain the difference in portfolio risk."""
        diff = quantum_risk - classical_risk
        
        if abs(diff) < 1:
            return "Both methods result in similar risk levels for your portfolio."
        elif diff < 0:
            return f"The quantum method reduces portfolio risk by {abs(diff):.2f}%. This means less volatility and more stable returns."
        else:
            return f"The quantum method has {diff:.2f}% higher risk. This means more volatility but potentially higher rewards."
    
    def _explain_sharpe_ratio(self, classical_sharpe: float, quantum_sharpe: float) -> str:
        """Explain Sharpe ratio in simple terms."""
        if classical_sharpe > quantum_sharpe:
            winner = "classical"
            better_ratio = classical_sharpe
        else:
            winner = "quantum"
            better_ratio = quantum_sharpe
        
        if better_ratio > 1.5:
            performance = "excellent"
        elif better_ratio > 1.0:
            performance = "good"
        elif better_ratio > 0.5:
            performance = "moderate"
        else:
            performance = "poor"
        
        return f"The {winner} method has {performance} risk-adjusted performance. Sharpe ratio measures how much extra return you get for the extra risk you take - higher is better."
    
    def _analyze_diversification(self, classical: Dict, quantum: Dict) -> Dict:
        """Analyze diversification differences."""
        classical_weights = classical.get('weights', [])
        quantum_weights = quantum.get('weights', [])
        
        # Count assets with meaningful allocation (>1%)
        classical_assets = np.sum(np.array(classical_weights) > 0.01)
        quantum_assets = np.sum(np.array(quantum_weights) > 0.01)
        
        # Calculate concentration (how spread out the investments are)
        classical_concentration = np.sum(np.array(classical_weights) ** 2) if len(classical_weights) > 0 else 1
        quantum_concentration = np.sum(np.array(quantum_weights) ** 2) if len(quantum_weights) > 0 else 1
        
        analysis = {
            'classical_assets': int(classical_assets),
            'quantum_assets': int(quantum_assets),
            'diversification_winner': 'Quantum' if quantum_assets > classical_assets else 'Classical',
            'explanation': self._explain_diversification(classical_assets, quantum_assets, classical_concentration, quantum_concentration)
        }
        
        return analysis
    
    def _explain_diversification(self, classical_assets: int, quantum_assets: int, 
                               classical_conc: float, quantum_conc: float) -> str:
        """Explain diversification in simple terms."""
        if classical_assets == quantum_assets:
            if classical_conc < quantum_conc:
                return f"Both methods invest in {classical_assets} assets, but classical spreads investments more evenly (better diversification)."
            elif quantum_conc < classical_conc:
                return f"Both methods invest in {classical_assets} assets, but quantum spreads investments more evenly (better diversification)."
            else:
                return f"Both methods invest in {classical_assets} assets with similar diversification."
        elif quantum_assets > classical_assets:
            return f"Quantum method invests in {quantum_assets} assets vs {classical_assets} for classical. More assets usually means better diversification and lower risk."
        else:
            return f"Classical method invests in {classical_assets} assets vs {quantum_assets} for quantum. More assets usually means better diversification and lower risk."
    
    def _get_practical_recommendation(self, classical: Dict, quantum: Dict) -> str:
        """Provide practical investment recommendation."""
        classical_sharpe = classical.get('sharpe_ratio', 0)
        quantum_sharpe = quantum.get('sharpe_ratio', 0)
        classical_risk = classical.get('risk', 0)
        quantum_risk = quantum.get('risk', 0)
        
        # Decision logic
        if abs(classical_sharpe - quantum_sharpe) < 0.1:
            if classical_risk < quantum_risk:
                return "💡 Recommendation: Go with the Classical method - similar returns but lower risk."
            else:
                return "💡 Recommendation: Go with the Quantum method - similar returns but lower risk."
        elif classical_sharpe > quantum_sharpe:
            return "💡 Recommendation: Classical method offers better risk-adjusted returns for your portfolio."
        else:
            return "💡 Recommendation: Quantum method offers better risk-adjusted returns for your portfolio."
    
    def _explain_computation_time(self, classical: Dict, quantum: Dict) -> Dict:
        """Explain computation time differences."""
        classical_time = classical.get('computation_time', 0)
        quantum_time = quantum.get('computation_time', 0)
        
        if classical_time > 0 and quantum_time > 0:
            if classical_time < quantum_time:
                faster = "Classical"
                speedup = quantum_time / classical_time
            else:
                faster = "Quantum"
                speedup = classical_time / quantum_time
            
            explanation = f"{faster} method was {speedup:.1f}x faster ({classical_time:.2f}s vs {quantum_time:.2f}s)"
        else:
            explanation = "Both methods completed quickly"
        
        return {
            'classical_time': f"{classical_time:.2f}s",
            'quantum_time': f"{quantum_time:.2f}s",
            'explanation': explanation
        }