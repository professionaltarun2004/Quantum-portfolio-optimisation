# Interactive Quantum vs. AI/ML Portfolio Optimization Web App

A full-stack web application that compares classical (AI/ML) and quantum (VQE/QAOA) optimization approaches for stock portfolio allocation, using real market data. The app enables interactive user exploration and educational visualization of both optimization paradigms.

## 🚀 Features

### Frontend (Streamlit)
- **Interactive Interface**: User-friendly web interface for parameter selection
- **Stock Selection**: Manual ticker entry or CSV file upload
- **Real-time Data**: Fetches live market data using Yahoo Finance
- **Tabbed Results**: Separate views for classical, quantum, and comparison results
- **Dynamic Visualizations**: Interactive charts using Plotly
- **Educational Content**: Comprehensive explanations and learning resources

### Backend Optimization
- **Classical Methods**: Mean-variance optimization with ML enhancements
- **Quantum Algorithms**: VQE and QAOA using Qiskit
- **QUBO Encoding**: Converts portfolio problems to quantum-compatible format
- **Performance Comparison**: Side-by-side analysis of both approaches
- **Educational Layer**: Automated insights and explanations

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd quantum-portfolio-optimizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

## 📊 Usage

### Basic Workflow
1. **Configure Parameters**: Use the sidebar to set up your optimization
   - Enter stock tickers (e.g., AAPL,GOOGL,MSFT,AMZN,TSLA)
   - Select time period for historical data
   - Adjust risk tolerance and portfolio constraints
   - Choose optimization methods (Classical, Quantum, or both)

2. **Run Optimization**: Click "Run Optimization" to start the process
   - The app fetches real market data
   - Runs selected optimization algorithms
   - Generates comprehensive results and visualizations

3. **Analyze Results**: Explore the tabbed interface
   - **Classical Results**: Traditional optimization outcomes
   - **Quantum Results**: Quantum algorithm performance
   - **Comparison**: Side-by-side analysis and metrics
   - **Explanations**: Educational content and insights

### Advanced Features
- **Quantum Parameters**: Adjust circuit depth and shots for quantum algorithms
- **ML Enhancement**: Enable machine learning features for classical optimization
- **Export Results**: Save optimization results and visualizations
- **Educational Mode**: Access detailed explanations and learning resources

## 🔬 Technical Details

### Classical Optimization
- **Mean-Variance Optimization**: Markowitz portfolio theory implementation
- **ML Enhancement**: Random Forest regression for return prediction
- **Constraints**: Long-only positions with concentration limits
- **Solver**: CVXPY for quadratic programming

### Quantum Optimization
- **VQE (Variational Quantum Eigensolver)**: Hybrid quantum-classical approach
- **QAOA (Quantum Approximate Optimization Algorithm)**: Combinatorial optimization
- **QUBO Encoding**: Quadratic Unconstrained Binary Optimization formulation
- **Backend**: Qiskit AerSimulator for quantum circuit simulation

### Performance Metrics
- **Expected Return**: Annualized portfolio return
- **Risk (Volatility)**: Portfolio standard deviation
- **Sharpe Ratio**: Risk-adjusted return measure
- **Diversification**: Portfolio concentration analysis
- **Computation Time**: Algorithm efficiency comparison

## 📚 Educational Content

The app includes comprehensive educational materials:

### Theoretical Background
- Modern Portfolio Theory fundamentals
- Quantum computing principles for optimization
- QUBO formulation and Ising models
- VQE and QAOA algorithm explanations

### Practical Insights
- Performance comparison analysis
- Allocation similarity metrics
- Computational efficiency discussion
- Technology readiness assessment

### Learning Resources
- Recommended books and research papers
- Online courses and tutorials
- Video explanations and demonstrations
- Interactive algorithm visualizations

## 🏗️ Architecture

```
quantum-portfolio-optimizer/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── backend/
│   ├── __init__.py
│   ├── classical_optimizer.py      # Classical optimization methods
│   ├── quantum_optimizer.py        # Quantum algorithms (VQE/QAOA)
│   ├── qubo_encoder.py            # QUBO problem encoding
│   ├── comparator.py              # Results comparison and analysis
│   └── explanation_layer.py       # Educational content generation
└── README.md                      # This file
```

## 🔧 Configuration

### Environment Variables
- No special environment variables required
- All configuration is done through the web interface

### Quantum Backend
- Default: Qiskit AerSimulator (local simulation)
- Can be extended to use IBM Quantum hardware
- Supports custom quantum backends

### Data Sources
- Yahoo Finance API for market data
- Configurable time periods (1y, 2y, 3y, 5y)
- Support for custom datasets via CSV upload

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### AWS Deployment (Recommended)
1. **EC2 Instance**: Deploy on AWS EC2 for compute
2. **App Runner**: Use AWS App Runner for managed deployment
3. **S3 Storage**: Store results and data in S3 buckets
4. **Lambda Functions**: Optional for async processing

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

### Performance Tests
```bash
python -m pytest tests/performance/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Qiskit Team**: For the excellent quantum computing framework
- **Streamlit Team**: For the intuitive web app framework
- **Yahoo Finance**: For providing free market data API
- **Research Community**: For advancing quantum optimization algorithms

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `/docs` folder
- Review the educational content in the app

## 🔮 Future Enhancements

### Planned Features
- **Multi-objective Optimization**: Pareto frontier analysis
- **Real-time Updates**: Live portfolio monitoring
- **Advanced Quantum Backends**: IBM Quantum hardware integration
- **Portfolio Backtesting**: Historical performance analysis
- **Risk Models**: Advanced risk factor modeling

### Research Directions
- **Hybrid Algorithms**: Classical-quantum hybrid approaches
- **Noise Mitigation**: Error correction for quantum algorithms
- **Scalability Studies**: Large portfolio optimization
- **Alternative Encodings**: Different QUBO formulations

---

**Built with ❤️ for the quantum computing and finance communities**