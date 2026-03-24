import numpy as np
from typing import Dict, Any, List

class RewardEngine:
    """ Computes and smooths the systemic reward signal. """
    def __init__(self, lambda_dd: float = 2.0, lambda_tx: float = 5.0, alpha: float = 0.2):
        self.lambda_dd = lambda_dd 
        self.lambda_tx = lambda_tx 
        self.alpha = alpha         
        self.smoothed_reward = 0.0
        
    def calculate_reward(self, sharpe: float, drawdown: float, tx_cost: float) -> float:
        """ 
        Reward = (Sharpe * 0.5) - (λ_1 * |Drawdown|) - (λ_2 * Transaction_Cost * 10)
        """
        raw_reward = (sharpe * 0.5) - (self.lambda_dd * abs(drawdown)) - (self.lambda_tx * tx_cost * 10)
        
        # 1. TEMPORAL REWARD SMOOTHING 
        # Smooth the unnormalized raw rewards FIRST to suppress localized outliers structurally
        if self.smoothed_reward == 0.0:
            self.smoothed_reward = raw_reward
        else:
            self.smoothed_reward = self.alpha * raw_reward + (1 - self.alpha) * self.smoothed_reward
            
        return float(self.smoothed_reward)

class ExperienceMemory:
    """ Stores sequential RL transition tuples and global reward distributions. """
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.buffer = []
        self.historical_rewards = []
        
    def add(self, state: np.ndarray, latent_mu: np.ndarray, latent_action: np.ndarray, reward: float):
        self.buffer.append((state, latent_mu, latent_action, reward))
        self.historical_rewards.append(reward)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
        if len(self.historical_rewards) > 1000:
            self.historical_rewards.pop(0)

    def get_all(self):
        return self.buffer
        
    def get_reward_stats(self):
        """ Returns running global mean and std for Reward Normalization bounds. """
        if len(self.historical_rewards) < 2:
            return 0.0, 1.0
        return np.mean(self.historical_rewards), np.std(self.historical_rewards) + 1e-8
        
    def clear(self):
        self.buffer = []

class RLPolicyEngine:
    """
    QRL-Lite Policy Engine (MLP-Based).
    Continuous Policy Gradient (REINFORCE) algorithm mapping continuous market states 
    to QUBO boundaries, utilizing a hidden layer, Action Bounding (Sigmoid Squashing),
    Advantage Baselines, and Gradient Batching.
    """
    def __init__(self, state_dim: int = 7, action_dim: int = 3, hidden_dim: int = 8, learning_rate: float = 0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.lr = learning_rate
        self.lr_decay = 0.999
        self.batch_size = 5  # Execute gradient descent after N collected experiences
        
        # Neural Weights (1 Hidden Layer MLP)
        np.random.seed(42) 
        # He initialization for Tanh boundaries
        self.W1 = np.random.randn(hidden_dim, state_dim) * np.sqrt(1.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # Xavier initialization mapping to latent bounds
        self.W2 = np.random.randn(action_dim, hidden_dim) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros(action_dim) 
        
        # Exploration Control (Sigma Decay)
        self.sigma = 0.4
        self.min_sigma = 0.05
        self.sigma_decay = 0.995
        
        self.memory = ExperienceMemory()
        self.reward_engine = RewardEngine()

        # Hard mathematical constraints for final QUBO scaling mappings
        self.action_bounds = [
            (0.1, 5.0),  # risk_aversion_q bounds
            (0.1, 2.0),  # constraint_scaling bounds
            (0.1, 5.0)   # diversification_weight bounds
        ]
        
        self._last_state = np.zeros(state_dim)
        self._last_latent_mu = np.zeros(action_dim)
        self._last_latent_action = np.zeros(action_dim)
        
        # Advantage Function Baseline
        self.baseline_reward = 0.0 
        self.baseline_alpha = 0.1

    def _extract_state_tensor(self, market_state: Dict[str, Any]) -> np.ndarray:
        metrics = market_state.get('metrics', {})
        
        vol = metrics.get('volatility', 0.15)
        mom = metrics.get('momentum', 0.0)
        dd = abs(metrics.get('drawdown', 0.0))
        ret = metrics.get('recent_return', 0.0)
        
        # Normalize continuous variables preventing internal MLP geometric explosions
        norm_vol = (vol - 0.15) / 0.10
        norm_mom = np.clip(mom * 5.0, -2.0, 2.0)
        norm_dd = np.clip(dd * 5.0, 0.0, 2.0)
        norm_ret = np.clip(ret * 10.0, -2.0, 2.0)
        
        # Preserve memory of prior state configurations internally
        prev = [0.0, 0.0, 0.0]
        
        return np.array([norm_vol, norm_mom, norm_dd, norm_ret, prev[0], prev[1], prev[2]])

    def get_action(self, market_state: Dict[str, Any], explore: bool = True) -> Dict[str, float]:
        """ 
        Forward Pass: Computes means, injects noise deeply into latent space, 
        then maps probabilistically using Sigmoid Action Bounding.
        """
        state_vec = self._extract_state_tensor(market_state)
        
        # Linear layer 1 -> Tanh Activation -> Linear layer 2
        A1 = np.tanh(np.dot(self.W1, state_vec) + self.b1)
        mu_raw = np.dot(self.W2, A1) + self.b2
        
        if explore:
            # Injecting noise directly into unbounded mathematical latent planes 
            # prevents clipping errors interrupting gradient chains
            latent_action = np.random.normal(mu_raw, self.sigma)
            self.sigma = max(self.min_sigma, self.sigma * self.sigma_decay)
        else:
            latent_action = mu_raw
            
        # 1-to-1 Sigmoid deterministic squashing guarantees safe parameters internally
        action_squashed = 1.0 / (1.0 + np.exp(-latent_action))
        
        # Scale structurally onto explicit mathematical boundaries required by QUBO
        action_scaled = np.zeros(self.action_dim)
        for i, bounds in enumerate(self.action_bounds):
            b_min, b_max = bounds
            action_scaled[i] = b_min + (b_max - b_min) * action_squashed[i]
        
        # Cache for localized Multi-Layer Backprop
        self._last_state = state_vec
        self._last_latent_mu = mu_raw
        self._last_latent_action = latent_action
        
        return {
            "risk_aversion_q": float(action_scaled[0]),
            "constraint_scaling": float(action_scaled[1]),
            "diversification_weight": float(action_scaled[2])
        }
        
    def store_experience(self, sharpe: float, drawdown: float, tx_cost: float):
        reward = self.reward_engine.calculate_reward(sharpe, drawdown, tx_cost)
        self.memory.add(self._last_state, self._last_latent_mu, self._last_latent_action, reward)
        
    def update_policy(self):
        """
        Backward Pass (Gradient Ascent):
        Deploys Batch-level Advantage-based Multi-Layer Backpropagation mapping
        financial feedback formally onto the hidden MLP layer representations.
        """
        experiences = self.memory.get_all()
        if len(experiences) < self.batch_size:
            return  # Execute Mini-Batch Updates exclusively
            
        mean_r, std_r = self.memory.get_reward_stats()
        
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        
        # Strict evaluation Order: Smoothing -> Normalization -> Baseline Subtraction
        for state, mu, latent_action, smooth_reward in experiences:
            # 2. SEQUENCE NORMALIZATION (applied to already smoothed rewards)
            r_norm = (smooth_reward - mean_r) / std_r
            
            # 3. BASELINE SUBTRACTION (The Advantage Function isolating outperformances)
            self.baseline_reward = (1 - self.baseline_alpha) * self.baseline_reward + self.baseline_alpha * r_norm
            advantage = r_norm - self.baseline_reward
            
            # Reconstruct transient forward pass activations needed algebraically for deep backprop
            Z1 = np.dot(self.W1, state) + self.b1
            A1 = np.tanh(Z1)
            
            # Policy Gradient w.r.t LATENT space ensures true gradient preservation under sigmoid squashes
            dlogpi = (latent_action - mu) / (self.sigma ** 2 + 1e-8)
            
            # Formally calculate Outer Product for Top Layer (W2)
            dmu = dlogpi * advantage
            db2 += dmu
            dW2 += np.outer(dmu, A1)
            
            # Calculate Tanh derivatives tracking down to Bottom Layer (W1)
            dA1 = np.dot(self.W2.T, dmu)
            dZ1 = dA1 * (1.0 - A1**2)
            db1 += dZ1
            dW1 += np.outer(dZ1, state)
            
        # Averages explicitly over empirical batch size
        dW1 /= len(experiences)
        db1 /= len(experiences)
        dW2 /= len(experiences)
        db2 /= len(experiences)
        
        # Strict Gradient Clipping mitigating cascading covariance blowouts
        clip_val = 1.0
        dW1 = np.clip(dW1, -clip_val, clip_val)
        db1 = np.clip(db1, -clip_val, clip_val)
        dW2 = np.clip(dW2, -clip_val, clip_val)
        db2 = np.clip(db2, -clip_val, clip_val)
        
        # Neural Update Trace
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
        
        self.lr *= self.lr_decay
        self.memory.clear()
