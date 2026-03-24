import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """Deep Q-Network untuk trading agent"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class TradingEnvironment:
    """Trading environment untuk Reinforcement Learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001, window: int = 20):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window = window
        
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = self.window
        self.balance = self.initial_balance
        self.position = 0  # Number of shares held
        self.trades = []
        self.equity_curve = [self.balance]
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        if self.current_step >= len(self.data):
            return None
        
        # Get price data for last window
        price_data = self.data.iloc[self.current_step - self.window:self.current_step]
        
        # Features: price, returns, volume, position, balance
        features = [
            price_data['Close'].values / price_data['Close'].mean() - 1,
            price_data['Close'].pct_change().values,
            price_data['Volume'].values / price_data['Volume'].mean(),
            np.full(self.window, self.position / self.initial_balance),
            np.full(self.window, self.balance / self.initial_balance)
        ]
        
        state = np.concatenate([f.flatten() for f in features])
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action: 0=hold, 1=buy, 2=sell
        """
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute trade
        if action == 1:  # Buy
            max_shares = int(self.balance / (current_price * (1 + self.transaction_cost)))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position += max_shares
                self.trades.append(('BUY', self.current_step, current_price, max_shares))
        
        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.trades.append(('SELL', self.current_step, current_price, self.position))
                self.position = 0
        
        # Calculate reward (profit change)
        current_value = self.balance + self.position * current_price
        previous_value = self.equity_curve[-1]
        reward = (current_value - previous_value) / previous_value
        
        # Update equity curve
        self.equity_curve.append(current_value)
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done, {'value': current_value, 'position': self.position}
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.equity_curve:
            return {}
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        return {
            'final_balance': self.equity_curve[-1],
            'total_return': (self.equity_curve[-1] - self.initial_balance) / self.initial_balance,
            'sharpe_ratio': sharpe,
            'num_trades': len(self.trades),
            'equity_curve': self.equity_curve
        }

class DQNTradingAgent:
    """Deep Q-Network Trading Agent"""
    
    def __init__(self, state_dim: int, action_dim: int = 3,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000,
                 batch_size: int = 64, target_update: int = 100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        logger.info(f"DQN Agent initialized on {self.device}")
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax().item()
    
    def replay(self) -> float:
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def train(self, env: TradingEnvironment, episodes: int = 100) -> Dict:
        """Train agent on environment"""
        scores = []
        losses = []
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.act(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.remember(state, action, reward, next_state, done)
                loss = self.replay()
                
                state = next_state
                total_reward += reward
                
                if loss > 0:
                    losses.append(loss)
            
            scores.append(total_reward)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward: {total_reward:.4f}, Epsilon: {self.epsilon:.4f}")
        
        metrics = env.get_metrics()
        return {
            'scores': scores,
            'losses': losses,
            'metrics': metrics,
            'final_epsilon': self.epsilon
        }
    
    def optimize_strategy(self, df: pd.DataFrame) -> Dict:
        """Optimize trading strategy using RL"""
        env = TradingEnvironment(df)
        
        # Get initial state dimension
        state = env.reset()
        state_dim = len(state)
        
        # Re-initialize with correct state dimension
        self.__init__(state_dim, self.action_dim)
        
        # Train
        results = self.train(env, episodes=50)
        
        return {
            'optimized_params': {
                'epsilon': results['final_epsilon'],
                'gamma': self.gamma
            },
            'performance': results['metrics'],
            'trades': env.trades
        }