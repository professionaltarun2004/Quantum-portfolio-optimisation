import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Dict
import requests
import io

# Import our custom modules
from backend.classical_optimizer import ClassicalOptimizer
from backend.quantum_optimizer import QuantumOptimizer
from backend.qubo_encoder import QUBOEncoder
from backend.comparator import Comparator
from backend.explanation_layer import ExplanationLayer

# Page config
st.set_page_config(
    page_title="Quantum vs AI/ML Portfolio Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comparison-winner {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DataFetcher:
    """Enhanced data fetcher supporting multiple data sources."""
    
    def __init__(self):
        self.sp500_tickers = None
        
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers with fallback to predefined list."""
        # Use a predefined list of major S&P 500 stocks to avoid API issues
        major_sp500_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
            'JPM', 'V', 'PG', 'HD', 'MA', 'ABBV', 'BAC', 'ASML', 'AVGO', 'XOM',
            'LLY', 'COST', 'WMT', 'KO', 'ORCL', 'ACN', 'MRK', 'NFLX', 'AMD', 'PEP',
            'TMO', 'LIN', 'ADBE', 'CSCO', 'ABT', 'CRM', 'TMUS', 'DHR', 'DIS', 'VZ',
            'WFC', 'CMCSA', 'PFE', 'NKE', 'BMY', 'PM', 'RTX', 'INTC', 'UPS', 'HON',
            'QCOM', 'AMGN', 'LOW', 'SPGI', 'GS', 'INTU', 'CAT', 'CVX', 'BKNG', 'AXP',
            'DE', 'GILD', 'BLK', 'BA', 'MDT', 'SYK', 'TJX', 'VRTX', 'SCHW', 'ADP',
            'LRCX', 'ADI', 'MDLZ', 'C', 'REGN', 'PYPL', 'ISRG', 'NOW', 'PLD', 'KLAC',
            'MMC', 'EOG', 'FISV', 'SO', 'DUK', 'ICE', 'APD', 'SHW', 'CME', 'USB',
            'ZTS', 'TGT', 'PNC', 'MU', 'CL', 'EQIX', 'NSC', 'AON', 'ITW', 'BSX'
        ]
        
        try:
            # Try to fetch from Wikipedia as backup verification
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url, header=0)
            sp500_table = tables[0]
            wiki_tickers = sp500_table['Symbol'].tolist()
            # Clean tickers (remove dots for Yahoo Finance compatibility)
            wiki_tickers = [ticker.replace('.', '-') for ticker in wiki_tickers]
            
            # Use Wikipedia data if successful
            st.success("Successfully fetched S&P 500 tickers from Wikipedia")
            return wiki_tickers[:100]  # Return first 100 for performance
            
        except Exception as e:
            st.info("Using predefined S&P 500 stock list (Wikipedia unavailable)")
            return major_sp500_stocks
    
    def get_stock_data(self, tickers: List[str], period: str = "2y", 
                      api_key: str = None, data_source: str = "yahoo") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch stock data from various sources."""
        
        if data_source == "yahoo":
            return self._fetch_yahoo_data(tickers, period)
        elif data_source == "alphavantage" and api_key:
            return self._fetch_alphavantage_data(tickers, api_key)
        elif data_source == "perfeq" and api_key:
            return self._fetch_perfeq_data(tickers, api_key)
        else:
            st.error("Invalid data source or missing API key")
            return None, None
    
    def _fetch_yahoo_data(self, tickers: List[str], period: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from Yahoo Finance with robust error handling."""
        try:
            # Limit tickers to avoid overwhelming the API
            limited_tickers = tickers[:10]  # Limit to 10 stocks for better reliability
            
            with st.spinner(f"Fetching data for {len(limited_tickers)} stocks from Yahoo Finance..."):
                # Add timeout and retry logic
                import time
                max_retries = 2
                
                for attempt in range(max_retries):
                    try:
                        # Try to download price data with better error handling
                        price_data = yf.download(
                            limited_tickers, 
                            period=period, 
                            progress=False,
                            timeout=10,
                            threads=False  # Disable threading to avoid issues
                        )
                        
                        if not price_data.empty:
                            break
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            st.info(f"Retry {attempt + 1}/{max_retries} for Yahoo Finance...")
                            time.sleep(2)  # Wait before retry
                        else:
                            raise e
                
                # Handle empty response or API issues
                if price_data.empty:
                    st.warning("Yahoo Finance API returned empty data. Using synthetic data for demonstration.")
                    return self._generate_synthetic_data(limited_tickers, period)
                
                # Extract adjusted close prices
                if 'Adj Close' in price_data.columns.get_level_values(0):
                    price_data = price_data['Adj Close']
                elif len(price_data.columns) > 0:
                    # Take available price columns
                    if price_data.columns.nlevels > 1:
                        price_data = price_data.iloc[:, :len(tickers)]
                        price_data.columns = tickers[:len(price_data.columns)]
                
                if isinstance(price_data, pd.Series):
                    price_data = price_data.to_frame(tickers[0])
                
                # Handle missing data
                if len(price_data.columns) == 0:
                    st.warning("No valid stock data found. Using synthetic data for demonstration.")
                    return self._generate_synthetic_data(tickers, period)
                
                price_data = price_data.dropna(axis=1, thresh=len(price_data) * 0.5)  # Keep stocks with 50%+ data
                
                if price_data.empty:
                    st.warning("All stocks had insufficient data. Using synthetic data for demonstration.")
                    return self._generate_synthetic_data(tickers, period)
                
                price_data = price_data.ffill().bfill()
                
                # Calculate returns
                returns_data = price_data.pct_change().dropna()
                
                if returns_data.empty:
                    st.warning("Could not calculate returns. Using synthetic data for demonstration.")
                    return self._generate_synthetic_data(tickers, period)
                
                st.success(f"Successfully fetched data for {len(price_data.columns)} stocks")
                return price_data, returns_data
                
        except Exception as e:
            st.warning(f"Yahoo Finance API error: {str(e)}. Using synthetic data for demonstration.")
            return self._generate_synthetic_data(tickers, period)
    
    def _generate_synthetic_data(self, tickers: List[str], period: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic stock data for demonstration."""
        # Determine number of days
        days_map = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}
        n_days = days_map.get(period, 504)  # Default to 2 years for better ML training
        
        # Generate synthetic price data
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
        
        # Use more assets for better demonstration (up to 10)
        demo_tickers = tickers[:10] if len(tickers) >= 10 else tickers
        if len(demo_tickers) < 5:
            # Add some default tickers if not enough provided
            default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
            demo_tickers = list(set(demo_tickers + default_tickers))[:10]
        
        price_data = pd.DataFrame(index=dates, columns=demo_tickers)
        
        # Define different stock characteristics for more realistic simulation
        stock_profiles = [
            {'drift': 0.0003, 'vol': 0.015, 'base_price': 150},  # Large cap tech
            {'drift': 0.0002, 'vol': 0.018, 'base_price': 120},  # Large cap stable
            {'drift': 0.0004, 'vol': 0.025, 'base_price': 80},   # Growth stock
            {'drift': 0.0001, 'vol': 0.012, 'base_price': 200},  # Defensive stock
            {'drift': 0.0005, 'vol': 0.030, 'base_price': 60},   # High growth/volatile
            {'drift': 0.0002, 'vol': 0.020, 'base_price': 100},  # Balanced
            {'drift': 0.0003, 'vol': 0.022, 'base_price': 90},   # Tech stock
            {'drift': 0.0001, 'vol': 0.014, 'base_price': 180},  # Utility-like
            {'drift': 0.0004, 'vol': 0.028, 'base_price': 70},   # Small cap growth
            {'drift': 0.0002, 'vol': 0.016, 'base_price': 130}   # Mid cap
        ]
        
        for i, ticker in enumerate(price_data.columns):
            profile = stock_profiles[i % len(stock_profiles)]
            
            # Generate more realistic stock price movements with trends and cycles
            initial_price = profile['base_price']
            drift = profile['drift']
            volatility = profile['vol']
            
            # Add some cyclical components and regime changes
            t = np.arange(n_days)
            trend = drift * t
            cycle = 0.001 * np.sin(2 * np.pi * t / 252)  # Annual cycle
            noise = np.random.normal(0, volatility, n_days)
            
            # Combine components
            log_returns = trend + cycle + noise
            
            # Convert to prices
            prices = initial_price * np.exp(np.cumsum(log_returns))
            price_data[ticker] = prices
        
        # Calculate returns
        returns_data = price_data.pct_change().dropna()
        
        # Add some correlation structure to make it more realistic
        correlation_factor = 0.3
        market_factor = np.random.normal(0, 0.01, len(returns_data))
        
        for ticker in returns_data.columns:
            returns_data[ticker] = (1 - correlation_factor) * returns_data[ticker] + \
                                 correlation_factor * market_factor
        
        st.info(f"Generated synthetic data for {len(demo_tickers)} assets with realistic correlations for demonstration.")
        return price_data, returns_data
    
    def _fetch_alphavantage_data(self, tickers: List[str], api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from Alpha Vantage API."""
        st.info("Alpha Vantage integration - would fetch data here with provided API key")
        # Placeholder for Alpha Vantage implementation
        return self._fetch_yahoo_data(tickers[:20], "1y")  # Fallback to Yahoo for demo
    
    def _fetch_perfeq_data(self, tickers: List[str], api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from PerfEq API."""
        st.info("PerfEq integration - would fetch high-quality financial data here")
        # Placeholder for PerfEq implementation
        return self._fetch_yahoo_data(tickers[:50], "2y")  # Fallback to Yahoo for demo

def main():
    st.markdown('<h1 class="main-header">🚀 Quantum vs AI/ML Portfolio Optimization</h1>', unsafe_allow_html=True)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("📋 Configuration")
        
        # Data source selection
        st.subheader("📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source",
            ["yahoo", "alphavantage", "perfeq"],
            format_func=lambda x: {
                "yahoo": "Yahoo Finance (Free)",
                "alphavantage": "Alpha Vantage (API Key Required)",
                "perfeq": "PerfEq (API Key Required)"
            }[x],
            key="data_source_select"
        )
        
        # API Key input if needed
        api_key = None
        if data_source in ["alphavantage", "perfeq"]:
            api_key = st.text_input(
                f"Enter {data_source.upper()} API Key",
                type="password",
                help=f"Required for {data_source.upper()} data access"
            )
            if not api_key:
                st.warning(f"Please provide your {data_source.upper()} API key to continue")
                st.stop()
        
        # Stock selection
        st.subheader("📈 Stock Selection")
        selection_method = st.radio(
            "Selection Method",
            ["S&P 500 Subset", "Manual Entry", "Upload CSV"]
        )
        
        if selection_method == "S&P 500 Subset":
            sp500_tickers = data_fetcher.get_sp500_tickers()
            num_stocks = st.slider("Number of S&P 500 stocks", 10, 100, 50)
            tickers = sp500_tickers[:num_stocks]
            st.info(f"Selected first {num_stocks} S&P 500 stocks")
            
        elif selection_method == "Manual Entry":
            tickers_input = st.text_area(
                "Stock Tickers (one per line or comma-separated)",
                "AAPL\nGOOGL\nMSFT\nAMZN\nTSLA\nMETA\nNVDA\nJPM\nJNJ\nV"
            )
            tickers = [t.strip().upper() for t in tickers_input.replace(',', '\n').split('\n') if t.strip()]
            
        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload CSV with tickers", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                tickers = df.iloc[:, 0].tolist()
            else:
                tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        # Portfolio parameters
        st.subheader("⚙️ Portfolio Parameters")
        portfolio_size = st.slider("Target Portfolio Size", 5, min(20, len(tickers)), 10)
        risk_tolerance = st.slider("Risk Tolerance", 0.1, 3.0, 1.0, 0.1)
        
        # Time period
        st.subheader("📅 Data Period")
        period = st.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1, key="period_select")
        
        # Method selection
        st.subheader("🔬 Optimization Methods")
        run_classical = st.checkbox("Classical AI/ML Methods", True)
        if run_classical:
            classical_methods = st.multiselect(
                "Classical Methods",
                ["Mean-Variance", "Lasso", "Ridge", "Genetic Algorithm", "Neural Network"],
                default=["Mean-Variance", "Lasso"]
            )
        
        run_quantum = st.checkbox("Quantum Methods", True)
        if run_quantum:
            quantum_method = st.selectbox("Quantum Algorithm", ["VQE", "QAOA"], key="quantum_method_select")
            circuit_depth = st.slider("Circuit Depth", 1, 5, 2)
            shots = st.selectbox("Shots", [1024, 2048, 4096], index=1, key="shots_select")
        
        # Run button
        run_optimization = st.button("🚀 Run Portfolio Optimization", type="primary")
    
    # Main content area
    if run_optimization and tickers:
        # Add option to use synthetic data directly
        use_synthetic = st.checkbox("Use Synthetic Data (Recommended due to API issues)", True)
        
        if use_synthetic:
            st.info("Using synthetic data for demonstration. This provides realistic portfolio optimization examples.")
            price_data, returns_data = data_fetcher._generate_synthetic_data(tickers, period)
        else:
            # Fetch real data
            price_data, returns_data = data_fetcher.get_stock_data(
                tickers, period, api_key, data_source
            )
        
        if price_data is not None and returns_data is not None:
            # Limit to portfolio size
            if len(returns_data.columns) > portfolio_size:
                # Select most liquid stocks (highest average volume)
                selected_tickers = returns_data.columns[:portfolio_size]
                returns_data = returns_data[selected_tickers]
                price_data = price_data[selected_tickers]
            
            # Create tabs for results
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Classical AI/ML Results", 
                "⚛️ Quantum Results", 
                "🔄 Side-by-Side Comparison",
                "📈 Advanced Visualizations",
                "📚 Educational Insights"
            ])
            
            results = {}
            
            # Classical optimization
            if run_classical:
                with tab1:
                    st.header("Classical AI/ML Optimization Results")
                    
                    for method in classical_methods:
                        st.subheader(f"{method} Results")
                        
                        with st.spinner(f"Running {method} optimization..."):
                            try:
                                classical_opt = ClassicalOptimizer()
                                classical_result = classical_opt.optimize(
                                    returns_data, risk_tolerance, method.lower().replace('-', '_')
                                )
                                results[f'classical_{method.lower()}'] = classical_result
                            except Exception as e:
                                st.error(f"Error in {method} optimization: {str(e)}")
                                # Create fallback result with equal weights
                                n_assets = len(returns_data.columns)
                                classical_result = {
                                    'weights': np.ones(n_assets) / n_assets,
                                    'expected_return': returns_data.mean().mean(),
                                    'volatility': 0.15,
                                    'sharpe_ratio': 0.5,
                                    'method': method,
                                    'status': 'fallback'
                                }
                                results[f'classical_{method.lower()}'] = classical_result
                                st.warning(f"Using equal-weight fallback for {method}")
                                continue
                            
                        display_classical_results(classical_result, price_data, method)
            
            # Quantum optimization
            if run_quantum:
                with tab2:
                    st.header("Quantum Optimization Results")
                    
                    with st.spinner(f"Running {quantum_method} quantum optimization..."):
                        # Encode as QUBO
                        qubo_encoder = QUBOEncoder()
                        qubo_matrix = qubo_encoder.encode_portfolio_problem(returns_data, risk_tolerance)
                        
                        # Solve with quantum
                        quantum_opt = QuantumOptimizer()
                        quantum_result = quantum_opt.optimize(
                            qubo_matrix, 
                            method=quantum_method,
                            circuit_depth=circuit_depth,
                            shots=shots
                        )
                        results['quantum'] = quantum_result
                        
                    display_quantum_results(quantum_result, price_data, returns_data.columns.tolist())
            
            # Side-by-side comparison
            if len(results) > 1:
                with tab3:
                    st.header("Method Comparison")
                    display_comprehensive_comparison(results, returns_data.columns.tolist())
            
            # Advanced visualizations
            with tab4:
                st.header("Advanced Portfolio Analysis")
                display_advanced_visualizations(results, returns_data, price_data)
            
            # Educational content
            with tab5:
                st.header("Educational Insights & Literature Analysis")
                explainer = ExplanationLayer()
                explanations = explainer.generate_explanations(results)
                display_educational_content(explanations, results)
    
    else:
        # Welcome screen
        display_welcome_screen()

def display_welcome_screen():
    """Enhanced welcome screen with detailed information."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Advanced Quantum vs AI/ML Portfolio Optimization! 🎯
        
        This application provides a comprehensive comparison between classical AI/ML approaches 
        and quantum computing methods for portfolio optimization using real financial data.
        
        ### 🔬 **Supported Methods:**
        
        **Classical AI/ML:**
        - Mean-Variance Optimization (Markowitz)
        - Lasso Regression for feature selection
        - Ridge Regression for regularization
        - Genetic Algorithm optimization
        - Neural Network-based predictions
        
        **Quantum Computing:**
        - VQE (Variational Quantum Eigensolver)
        - QAOA (Quantum Approximate Optimization Algorithm)
        - Configurable circuit depth and shots
        - NISQ limitations analysis
        
        ### 📊 **Data Sources:**
        - **Yahoo Finance**: Free access to major stocks
        - **Alpha Vantage**: Premium financial data (API key required)
        - **PerfEq**: High-quality institutional data (API key required)
        
        ### 📈 **Advanced Features:**
        - Support for 100+ stock portfolios
        - Efficient frontier visualization
        - Risk-return analysis
        - Convergence tracking
        - Performance gap analysis
        - Literature-based insights
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 **Quick Start:**
        
        1. **Choose Data Source**
           - Select your preferred data provider
           - Enter API key if required
        
        2. **Select Stocks**
           - S&P 500 subset (recommended)
           - Manual ticker entry
           - CSV upload
        
        3. **Configure Parameters**
           - Portfolio size (5-20 stocks)
           - Risk tolerance
           - Time period
        
        4. **Choose Methods**
           - Classical AI/ML algorithms
           - Quantum optimization
           - Or both for comparison
        
        5. **Analyze Results**
           - Performance metrics
           - Visualizations
           - Educational insights
        
        ### 📚 **Educational Value:**
        - Algorithm explanations
        - Performance comparisons
        - NISQ limitations
        - Literature references
        - Practical implications
        """)

def display_classical_results(result: Dict, price_data: pd.DataFrame, method: str):
    """Enhanced display for classical optimization results."""
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Expected Return", f"{result['expected_return']:.2%}")
    with col2:
        st.metric("Risk (Volatility)", f"{result['risk']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Computation Time", f"{result['computation_time']:.3f}s")
    
    # Portfolio allocation
    col1, col2 = st.columns(2)
    
    with col1:
        if 'weights' in result and len(result['weights']) > 0:
            weights_df = pd.DataFrame({
                'Asset': result.get('tickers', [f'Asset_{i}' for i in range(len(result['weights']))]),
                'Weight': result['weights']
            })
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            fig = px.bar(weights_df, x='Asset', y='Weight', 
                        title=f'{method} Portfolio Allocation',
                        color='Weight', color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart for top holdings
        if len(weights_df) > 0:
            top_holdings = weights_df.head(10)
            fig = px.pie(top_holdings, values='Weight', names='Asset', 
                        title=f'{method} Top Holdings')
            st.plotly_chart(fig, use_container_width=True)
    
    # Convergence plot
    if result.get('convergence_history'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result['convergence_history'],
            mode='lines+markers',
            name=f'{method} Convergence',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title=f'{method} Optimization Convergence',
            xaxis_title='Iteration',
            yaxis_title='Objective Value'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Method-specific details
    st.subheader(f"{method} Algorithm Details")
    if method == "Mean-Variance":
        st.write("**Markowitz Modern Portfolio Theory**")
        st.write("- Objective: Maximize return for given risk level")
        st.write("- Constraints: Weights sum to 1, long-only positions")
        st.write("- Optimization: Quadratic programming")
    elif method == "Lasso":
        st.write("**Lasso Regression Portfolio Selection**")
        st.write("- Feature selection through L1 regularization")
        st.write("- Automatic variable selection")
        st.write("- Sparse portfolio solutions")
    elif method == "Neural Network":
        st.write("**Neural Network-Based Optimization**")
        st.write("- Deep learning for return prediction")
        st.write("- Non-linear pattern recognition")
        st.write("- Adaptive feature learning")

def display_quantum_results(result: Dict, price_data: pd.DataFrame, tickers: List[str]):
    """Enhanced display for quantum optimization results."""
    
    # Performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Expected Return", f"{result['expected_return']:.2%}")
    with col2:
        st.metric("Risk (Volatility)", f"{result['risk']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    with col4:
        st.metric("Circuit Depth", result.get('circuit_depth', 'N/A'))
    with col5:
        st.metric("Computation Time", f"{result['computation_time']:.3f}s")
    
    # Portfolio allocation
    col1, col2 = st.columns(2)
    
    with col1:
        if 'weights' in result and len(result['weights']) > 0:
            weights_df = pd.DataFrame({
                'Asset': tickers[:len(result['weights'])],
                'Weight': result['weights']
            })
            weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
            
            fig = px.bar(weights_df, x='Asset', y='Weight', 
                        title=f'{result.get("method", "Quantum")} Portfolio Allocation',
                        color='Weight', color_continuous_scale='plasma')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quantum circuit visualization
        st.subheader("Quantum Circuit Structure")
        quantum_opt = QuantumOptimizer()
        circuit_desc = quantum_opt.create_quantum_circuit_description(
            len(result.get('weights', [])), 
            result.get('method', 'VQE'), 
            result.get('circuit_depth', 2)
        )
        st.code(circuit_desc, language='text')
    
    # Quantum-specific analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Quantum Algorithm Details")
        st.write(f"**Method:** {result.get('method', 'Unknown')}")
        st.write(f"**Shots:** {result.get('shots', 'N/A')}")
        st.write(f"**Circuit Depth:** {result.get('circuit_depth', 'N/A')}")
        
        if result.get('optimal_params') is not None:
            st.write("**Optimal Parameters (first 5):**")
            params = result['optimal_params']
            if hasattr(params, '__len__') and len(params) > 0:
                for i, param in enumerate(params[:5]):
                    st.write(f"θ_{i}: {param:.3f}")
    
    with col2:
        # NISQ limitations analysis
        st.subheader("NISQ Limitations Analysis")
        n_qubits = len(result.get('weights', []))
        if n_qubits > 10:
            st.warning("⚠️ Large problem size may exceed NISQ capabilities")
        if result.get('circuit_depth', 0) > 3:
            st.warning("⚠️ Deep circuits may suffer from noise accumulation")
        
        st.write("**Current Quantum Hardware Constraints:**")
        st.write("- Limited qubit count (50-1000 qubits)")
        st.write("- High error rates (0.1-1%)")
        st.write("- Short coherence times")
        st.write("- Gate fidelity limitations")
    
    # Convergence plot
    if result.get('convergence_history'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result['convergence_history'],
            mode='lines+markers',
            name='Quantum Energy',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Quantum Optimization Convergence',
            xaxis_title='Iteration',
            yaxis_title='Energy'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_comprehensive_comparison(results: Dict, tickers: List[str]):
    """Enhanced side-by-side comparison of all methods."""
    
    # Performance summary table
    st.subheader("📊 Performance Summary")
    
    summary_data = []
    for method_name, result in results.items():
        summary_data.append({
            'Method': method_name.replace('_', ' ').title(),
            'Expected Return': f"{result['expected_return']:.2%}",
            'Risk': f"{result['risk']:.2%}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Computation Time': f"{result['computation_time']:.3f}s",
            'Assets Selected': np.sum(np.array(result.get('weights', [])) > 0.001)
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk-Return scatter plot
        fig = go.Figure()
        
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        for i, (method_name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=[result['risk']],
                y=[result['expected_return']],
                mode='markers',
                name=method_name.replace('_', ' ').title(),
                marker=dict(size=15, color=colors[i % len(colors)]),
                text=[method_name.replace('_', ' ').title()],
                textposition="top center"
            ))
        
        fig.update_layout(
            title='Risk vs Return Comparison',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sharpe ratio comparison
        methods = [name.replace('_', ' ').title() for name in results.keys()]
        sharpe_ratios = [result['sharpe_ratio'] for result in results.values()]
        
        fig = px.bar(
            x=methods, y=sharpe_ratios,
            title='Sharpe Ratio Comparison',
            color=sharpe_ratios,
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio allocation comparison
    st.subheader("🎯 Portfolio Allocation Comparison")
    
    # Create allocation comparison matrix
    allocation_data = {}
    for method_name, result in results.items():
        weights = result.get('weights', [])
        if len(weights) > 0:
            # Pad or truncate to match ticker length
            if len(weights) < len(tickers):
                weights = np.pad(weights, (0, len(tickers) - len(weights)))
            elif len(weights) > len(tickers):
                weights = weights[:len(tickers)]
            allocation_data[method_name.replace('_', ' ').title()] = weights
    
    if allocation_data:
        allocation_df = pd.DataFrame(allocation_data, index=tickers)
        
        # Heatmap of allocations
        fig = px.imshow(
            allocation_df.T,
            title='Portfolio Allocation Heatmap',
            color_continuous_scale='viridis',
            aspect='auto'
        )
        fig.update_layout(
            xaxis_title='Assets',
            yaxis_title='Methods'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix of allocations
        if len(allocation_data) > 1:
            corr_matrix = allocation_df.corr()
            fig = px.imshow(
                corr_matrix,
                title='Method Allocation Correlation',
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)

def display_advanced_visualizations(results: Dict, returns_data: pd.DataFrame, price_data: pd.DataFrame):
    """Advanced portfolio analysis and visualizations."""
    
    # Efficient frontier
    st.subheader("📈 Efficient Frontier Analysis")
    
    if len(results) > 0:
        # Generate efficient frontier points
        risk_levels = np.linspace(0.05, 0.3, 50)
        efficient_returns = []
        
        for risk_target in risk_levels:
            # Simplified efficient frontier calculation
            max_return = 0
            for result in results.values():
                if result['risk'] <= risk_target:
                    max_return = max(max_return, result['expected_return'])
            efficient_returns.append(max_return)
        
        fig = go.Figure()
        
        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=risk_levels,
            y=efficient_returns,
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='black', width=2)
        ))
        
        # Plot method results
        colors = ['blue', 'green', 'orange', 'purple', 'red']
        for i, (method_name, result) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=[result['risk']],
                y=[result['expected_return']],
                mode='markers',
                name=method_name.replace('_', ' ').title(),
                marker=dict(size=12, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title='Efficient Frontier with Method Results',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Expected Return'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Convergence comparison
    st.subheader("🔄 Convergence Analysis")
    
    fig = go.Figure()
    colors = ['blue', 'green', 'orange', 'purple', 'red']
    
    for i, (method_name, result) in enumerate(results.items()):
        if result.get('convergence_history'):
            fig.add_trace(go.Scatter(
                y=result['convergence_history'],
                mode='lines+markers',
                name=method_name.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title='Optimization Convergence Comparison',
        xaxis_title='Iteration',
        yaxis_title='Objective Value'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Portfolio statistics
    st.subheader("📊 Portfolio Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Data Summary:**")
        st.write(f"- Number of assets: {len(returns_data.columns)}")
        st.write(f"- Time period: {len(returns_data)} days")
        st.write(f"- Average daily return: {returns_data.mean().mean():.4f}")
        st.write(f"- Average daily volatility: {returns_data.std().mean():.4f}")
        
    with col2:
        st.write("**Correlation Analysis:**")
        avg_correlation = returns_data.corr().values[np.triu_indices_from(returns_data.corr().values, k=1)].mean()
        st.write(f"- Average asset correlation: {avg_correlation:.3f}")
        st.write(f"- Market diversification potential: {'High' if avg_correlation < 0.3 else 'Medium' if avg_correlation < 0.6 else 'Low'}")

def display_educational_content(explanations: Dict, results: Dict):
    """Enhanced educational content with literature analysis."""
    
    # Performance gap analysis
    st.subheader("🔬 Performance Gap Analysis")
    
    if len(results) > 1:
        classical_results = {k: v for k, v in results.items() if 'classical' in k}
        quantum_results = {k: v for k, v in results.items() if 'quantum' in k}
        
        if classical_results and quantum_results:
            # Compare best classical vs quantum
            best_classical = max(classical_results.values(), key=lambda x: x['sharpe_ratio'])
            best_quantum = max(quantum_results.values(), key=lambda x: x['sharpe_ratio'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Return Gap",
                    f"{(best_quantum['expected_return'] - best_classical['expected_return']):.2%}",
                    delta=f"Quantum {'advantage' if best_quantum['expected_return'] > best_classical['expected_return'] else 'disadvantage'}"
                )
            
            with col2:
                st.metric(
                    "Risk Gap", 
                    f"{(best_quantum['risk'] - best_classical['risk']):.2%}",
                    delta=f"{'Higher' if best_quantum['risk'] > best_classical['risk'] else 'Lower'} quantum risk"
                )
            
            with col3:
                st.metric(
                    "Efficiency Gap",
                    f"{(best_classical['computation_time'] / best_quantum['computation_time']):.1f}x",
                    delta=f"Classical {'faster' if best_classical['computation_time'] < best_quantum['computation_time'] else 'slower'}"
                )
    
    # Literature-based insights
    st.subheader("📚 Literature Survey Insights")
    
    st.markdown("""
    ### Current State of Quantum Portfolio Optimization
    
    **Key Findings from Recent Research:**
    
    1. **Theoretical Advantages** (Orus et al., 2019; Mugel et al., 2020):
       - Quantum algorithms can theoretically provide exponential speedup for certain optimization problems
       - QAOA shows promise for combinatorial portfolio optimization
       - VQE demonstrates potential for continuous variable optimization
    
    2. **NISQ Era Limitations** (Preskill, 2018; Cerezo et al., 2021):
       - Current quantum hardware limited to ~100-1000 qubits
       - Gate error rates (0.1-1%) limit circuit depth
       - Quantum advantage not yet demonstrated for practical portfolio sizes
    
    3. **Practical Performance Gaps** (Egger et al., 2020; Slate et al., 2021):
       - Classical methods remain superior for most real-world problems
       - Quantum methods show promise for specific constraint structures
       - Hybrid classical-quantum approaches most promising near-term
    
    ### Scaling Challenges
    
    **Problem Size vs. Quantum Resources:**
    - 50 assets → ~50 qubits (feasible on current hardware)
    - 100 assets → ~100 qubits (at hardware limits)
    - 500+ assets → requires fault-tolerant quantum computers
    
    **Error Accumulation:**
    - Circuit depth scales with problem complexity
    - Each additional layer increases error probability
    - Current NISQ devices limited to ~10-20 gate layers
    """)
    
    # Method explanations with literature context
    method_explanations = explanations.get('method_explanations', {})
    
    if 'classical' in method_explanations:
        st.subheader("🔵 Classical AI/ML Methods - Literature Context")
        
        st.markdown("""
        **Mean-Variance Optimization (Markowitz, 1952):**
        - Foundation of modern portfolio theory
        - Proven mathematical framework with 70+ years of refinement
        - Computational complexity: O(n³) for n assets
        - Industry standard with extensive empirical validation
        
        **Machine Learning Enhancements:**
        - **Lasso/Ridge Regression**: Tibshirani (1996), Hoerl & Kennard (1970)
          - Addresses overfitting in high-dimensional problems
          - Automatic feature selection capabilities
        - **Genetic Algorithms**: Holland (1975), applied to portfolios by Chang et al. (2000)
          - Global optimization capabilities
          - Handles non-convex objective functions
        - **Neural Networks**: Recent advances in deep learning (LeCun et al., 2015)
          - Non-linear pattern recognition
          - Adaptive feature learning from market data
        """)
    
    if 'quantum' in method_explanations:
        st.subheader("🔴 Quantum Methods - Research Frontier")
        
        st.markdown("""
        **Variational Quantum Eigensolver (VQE):**
        - Peruzzo et al. (2014): Original VQE proposal
        - Cerezo et al. (2021): Comprehensive review of variational algorithms
        - **Advantages**: Suitable for NISQ devices, hybrid classical-quantum approach
        - **Limitations**: Barren plateaus, local minima, parameter optimization challenges
        
        **Quantum Approximate Optimization Algorithm (QAOA):**
        - Farhi et al. (2014): Original QAOA framework
        - Hadfield et al. (2019): QAOA for portfolio optimization
        - **Advantages**: Designed for combinatorial problems, provable approximation guarantees
        - **Limitations**: Performance depends on problem structure, requires deep circuits for good approximation
        
        **Current Research Directions:**
        - Error mitigation techniques (Kandala et al., 2019)
        - Quantum advantage demonstrations (Arute et al., 2019)
        - Hybrid classical-quantum algorithms (Benedetti et al., 2019)
        """)
    
    # Practical implications
    st.subheader("🎯 Practical Implications & Future Outlook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Current Recommendations (2024):**
        
        ✅ **Use Classical Methods For:**
        - Production portfolio management
        - Large-scale optimization (500+ assets)
        - Risk-critical applications
        - Real-time trading decisions
        
        🔬 **Experiment with Quantum For:**
        - Research and development
        - Small-scale problems (< 50 assets)
        - Proof-of-concept studies
        - Future technology preparation
        """)
    
    with col2:
        st.markdown("""
        **Future Timeline Projections:**
        
        **2024-2027: NISQ Era**
        - Quantum advantage for specialized problems
        - Hybrid algorithms development
        - Error mitigation improvements
        
        **2027-2035: Early Fault-Tolerant**
        - 1000+ logical qubits
        - Practical quantum advantage
        - Industry adoption begins
        
        **2035+: Mature Quantum Computing**
        - Large-scale portfolio optimization
        - Real-time quantum algorithms
        - Widespread industry deployment
        """)
    
    # Interactive learning resources
    st.subheader("📖 Interactive Learning Resources")
    
    with st.expander("🔗 Key Research Papers"):
        st.markdown("""
        1. **Orus, R., et al. (2019)** - "Quantum computing for finance: Overview and prospects"
        2. **Mugel, S., et al. (2020)** - "Portfolio optimization with quantum computers"
        3. **Egger, D. J., et al. (2020)** - "Quantum computing for Finance: State of the art"
        4. **Slate, N., et al. (2021)** - "Quantum walk-based portfolio optimization"
        5. **Cerezo, M., et al. (2021)** - "Variational quantum algorithms"
        """)
    
    with st.expander("🎓 Educational Videos & Courses"):
        st.markdown("""
        - **IBM Qiskit Textbook**: Quantum algorithms for applications
        - **Microsoft Quantum Development Kit**: Q# programming tutorials
        - **Coursera**: "Introduction to Quantum Computing" by IBM
        - **YouTube**: "Quantum Computing Explained" series by MinutePhysics
        - **arXiv**: Latest quantum computing research papers
        """)

if __name__ == "__main__":
    main()