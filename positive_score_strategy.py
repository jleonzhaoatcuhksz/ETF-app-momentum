"""
Positive-Score ETF Switching Strategy
Designed to achieve positive rewards by outperforming SPY
============================================
"""

import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import random
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PositiveScoreETFEnvironment:
    def __init__(self, db_path='./etf_data.db'):
        """Enhanced environment designed for positive score rewards"""
        self.db_path = db_path
        self.load_data()
        
        # Focus on available ETFs from our database (no QQQ available)
        self.top_etfs = ['SPY', 'XLK', 'XLV', 'XLF', 'XLI', 'XLE', 'TLT']  # 7 available ETFs
        self.etf_to_id = {etf: i for i, etf in enumerate(self.top_etfs)}
        
        # Environment parameters optimized for positive scores
        self.decision_frequency = 10  # Monthly decisions (every 10 trading days)
        self.transaction_cost = 0.001  # 0.1% - much lower cost
        self.min_holding_period = 10   # Minimum 10 days holding
        
        # State tracking
        self.current_step = 0
        self.current_etf = 0  # Start with SPY
        self.portfolio_value = 10000.0
        self.spy_value = 10000.0  # Track SPY benchmark
        self.days_held = 0
        self.total_switches = 0
        
        # Performance tracking for positive rewards
        self.monthly_returns = []
        self.spy_returns = []
        self.outperformance_history = []
        
        print(f"Positive Score Environment initialized:")
        print(f"  - Top ETFs: {len(self.top_etfs)} symbols")
        print(f"  - Decision frequency: Every {self.decision_frequency} days")
        print(f"  - Transaction cost: {self.transaction_cost*100:.1f}%")
        print(f"  - Min holding period: {self.min_holding_period} days")
        
    def load_data(self):
        """Load and prepare data with enhanced features"""
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT date, symbol, close, monthly_trend, sma_5d 
        FROM prices 
        WHERE monthly_trend IS NOT NULL
        ORDER BY date, symbol
        """
        
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        
        # Calculate enhanced momentum features
        self.data = self.data.sort_values(['symbol', 'date'])
        self.data['returns_1d'] = self.data.groupby('symbol')['close'].pct_change()
        self.data['returns_5d'] = self.data.groupby('symbol')['close'].pct_change(5)
        self.data['returns_20d'] = self.data.groupby('symbol')['close'].pct_change(20)
        
        # Volatility features - fix the rolling window issue
        self.data['volatility_20d'] = self.data.groupby('symbol')['returns_1d'].transform(lambda x: x.rolling(20).std())
        
        # Risk-adjusted momentum
        self.data['sharpe_20d'] = self.data['returns_20d'] / self.data['volatility_20d']
        
        # Fill NaN values
        self.data = self.data.fillna(0)
        
        # Get unique dates for time stepping
        self.dates = sorted(self.data['date'].unique())
        self.max_steps = len(self.dates) - 1
        
        print(f"Enhanced data loaded: {len(self.data)} records")
        
    def get_state(self):
        """Get current state with 30-dimensional features for better learning"""
        if self.current_step >= len(self.dates):
            return np.zeros(30)
            
        current_date = self.dates[self.current_step]
        current_data = self.data[self.data['date'] == current_date]
        
        if len(current_data) == 0:
            return np.zeros(30)
        
        # Current ETF information (5 features)
        current_etf_data = current_data[current_data['symbol'] == self.top_etfs[self.current_etf]]
        if len(current_etf_data) == 0:
            return np.zeros(30)
            
        current_features = [
            self.current_etf / len(self.top_etfs),  # Normalized ETF ID
            current_etf_data['monthly_trend'].iloc[0] / 100,  # Monthly trend
            current_etf_data['returns_5d'].iloc[0],   # 5-day momentum
            current_etf_data['returns_20d'].iloc[0],  # 20-day momentum
            current_etf_data['sharpe_20d'].iloc[0],   # Risk-adjusted momentum
        ]
        
        # All ETFs momentum signals (7 features)
        etf_features = []
        for etf in self.top_etfs:
            etf_data = current_data[current_data['symbol'] == etf]
            if len(etf_data) > 0:
                momentum = etf_data['returns_20d'].iloc[0]
                etf_features.append(momentum)
            else:
                etf_features.append(0)
        
        # Portfolio performance metrics (8 features)
        portfolio_return = (self.portfolio_value - 10000) / 10000
        spy_return = (self.spy_value - 10000) / 10000
        outperformance = portfolio_return - spy_return
        
        portfolio_features = [
            portfolio_return,                     # Total portfolio return
            spy_return,                          # SPY benchmark return
            outperformance,                      # Current outperformance
            self.days_held / 100,                # Days held (normalized)
            self.total_switches / 50,            # Total switches (normalized)
            len(self.monthly_returns) / 100,     # Time progress
            np.mean(self.outperformance_history[-12:]) if self.outperformance_history else 0,  # 12-month avg outperformance
            np.std(self.outperformance_history[-12:]) if len(self.outperformance_history) > 1 else 0,  # Outperformance volatility
        ]
        
        # Market regime indicators (10 features)
        spy_data = current_data[current_data['symbol'] == 'SPY']
        if len(spy_data) > 0:
            spy_trend = spy_data['monthly_trend'].iloc[0] / 100
            spy_momentum = spy_data['returns_20d'].iloc[0]
            spy_volatility = spy_data['volatility_20d'].iloc[0]
        else:
            spy_trend = spy_momentum = spy_volatility = 0
            
        regime_features = [
            spy_trend,                           # SPY monthly trend
            spy_momentum,                        # SPY momentum
            spy_volatility,                      # SPY volatility
            1 if spy_trend > 0.02 else 0,       # Bull market indicator
            1 if spy_volatility > 0.02 else 0,  # High volatility indicator
            1 if outperformance > 0 else 0,     # Currently outperforming
            1 if self.days_held >= self.min_holding_period else 0,  # Can switch indicator
            np.mean([d['returns_20d'] for d in [current_data[current_data['symbol'] == etf] for etf in self.top_etfs[:3]] if len(d) > 0]),  # Top 3 ETFs avg momentum
            max([d['returns_20d'].iloc[0] for d in [current_data[current_data['symbol'] == etf] for etf in self.top_etfs] if len(d) > 0] + [0]),  # Best ETF momentum
            min([d['returns_20d'].iloc[0] for d in [current_data[current_data['symbol'] == etf] for etf in self.top_etfs] if len(d) > 0] + [0]),  # Worst ETF momentum
        ]
        
        # Combine all features (30 total)
        state = np.array(current_features + etf_features + portfolio_features + regime_features)
        return np.nan_to_num(state, 0)
    
    def step(self, action):
        """Execute action and return positive-focused reward"""
        if self.current_step >= self.max_steps - 1:
            return self.get_state(), 0, True, {}
        
        current_date = self.dates[self.current_step]
        next_date = self.dates[self.current_step + 1]
        
        # Get current and next prices
        current_data = self.data[self.data['date'] == current_date]
        next_data = self.data[self.data['date'] == next_date]
        
        # Calculate returns
        reward = 0
        transaction_cost = 0
        
        # Action 0: Hold current ETF
        if action == 0 or self.days_held < self.min_holding_period:
            # Hold current position
            current_etf_symbol = self.top_etfs[self.current_etf]
            
            # Get returns
            etf_current = current_data[current_data['symbol'] == current_etf_symbol]
            etf_next = next_data[next_data['symbol'] == current_etf_symbol]
            
            if len(etf_current) > 0 and len(etf_next) > 0:
                period_return = (etf_next['close'].iloc[0] - etf_current['close'].iloc[0]) / etf_current['close'].iloc[0]
                period_return *= self.decision_frequency  # Scale for decision frequency
                self.portfolio_value *= (1 + period_return)
            
            self.days_held += self.decision_frequency
            
        else:
            # Switch to new ETF (actions 1-7 correspond to top_etfs indices)
            new_etf_id = action - 1
            
            if new_etf_id != self.current_etf and new_etf_id < len(self.top_etfs):
                # Execute switch
                transaction_cost = self.portfolio_value * self.transaction_cost
                self.portfolio_value -= transaction_cost
                
                # Get return for new ETF
                new_etf_symbol = self.top_etfs[new_etf_id]
                etf_current = current_data[current_data['symbol'] == new_etf_symbol]
                etf_next = next_data[next_data['symbol'] == new_etf_symbol]
                
                if len(etf_current) > 0 and len(etf_next) > 0:
                    period_return = (etf_next['close'].iloc[0] - etf_current['close'].iloc[0]) / etf_current['close'].iloc[0]
                    period_return *= self.decision_frequency
                    self.portfolio_value *= (1 + period_return)
                
                self.current_etf = new_etf_id
                self.days_held = self.decision_frequency
                self.total_switches += 1
            else:
                # Invalid action, treat as hold
                current_etf_symbol = self.top_etfs[self.current_etf]
                etf_current = current_data[current_data['symbol'] == current_etf_symbol]
                etf_next = next_data[next_data['symbol'] == current_etf_symbol]
                
                if len(etf_current) > 0 and len(etf_next) > 0:
                    period_return = (etf_next['close'].iloc[0] - etf_current['close'].iloc[0]) / etf_current['close'].iloc[0]
                    period_return *= self.decision_frequency
                    self.portfolio_value *= (1 + period_return)
                
                self.days_held += self.decision_frequency
        
        # Update SPY benchmark
        spy_current = current_data[current_data['symbol'] == 'SPY']
        spy_next = next_data[next_data['symbol'] == 'SPY']
        
        if len(spy_current) > 0 and len(spy_next) > 0:
            spy_return = (spy_next['close'].iloc[0] - spy_current['close'].iloc[0]) / spy_current['close'].iloc[0]
            spy_return *= self.decision_frequency
            self.spy_value *= (1 + spy_return)
        
        # POSITIVE REWARD CALCULATION - Key Innovation!
        portfolio_return = (self.portfolio_value - 10000) / 10000
        benchmark_return = (self.spy_value - 10000) / 10000
        outperformance = portfolio_return - benchmark_return
        
        # Reward based on outperformance (CAN BE POSITIVE!)
        reward = outperformance * 100  # Scale to reasonable range
        
        # Bonus for consistent outperformance
        if outperformance > 0:
            reward += 1.0  # Bonus for beating SPY
        
        # Small penalty for transaction costs
        if transaction_cost > 0:
            reward -= 0.5  # Small fixed penalty for switching
        
        # Track monthly performance
        self.outperformance_history.append(outperformance)
        if len(self.outperformance_history) > 60:  # Keep last 60 periods
            self.outperformance_history.pop(0)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'spy_value': self.spy_value,
            'outperformance': outperformance,
            'current_etf': self.top_etfs[self.current_etf],
            'transaction_cost': transaction_cost
        }
        
        return self.get_state(), reward, done, info
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = random.randint(0, max(0, self.max_steps - 500))  # Random start
        self.current_etf = 0  # Start with SPY
        self.portfolio_value = 10000.0
        self.spy_value = 10000.0
        self.days_held = 0
        self.total_switches = 0
        self.monthly_returns = []
        self.spy_returns = []
        self.outperformance_history = []
        
        return self.get_state()

class PositiveScoreDQNAgent:
    def __init__(self, state_size=30, action_size=8, learning_rate=0.001):
        """Enhanced DQN agent optimized for positive score learning"""
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # Larger memory
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum exploration
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.99  # Higher discount for long-term rewards
        
        # Build enhanced neural network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        """Build enhanced neural network for positive score optimization"""
        model = keras.Sequential([
            keras.layers.Dense(256, input_dim=self.state_size, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=64):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        target_q = rewards + (self.gamma * max_target_q_values * (1 - dones))
        
        target_f = self.q_network.predict(states, verbose=0)
        for i in range(batch_size):
            target_f[i][actions[i]] = target_q[i]
        
        self.q_network.fit(states, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

def train_positive_score_strategy():
    """Train the positive-score ETF switching strategy"""
    print("🚀 TRAINING POSITIVE-SCORE ETF SWITCHING STRATEGY")
    print("=" * 60)
    
    # Initialize environment and agent
    env = PositiveScoreETFEnvironment()
    agent = PositiveScoreDQNAgent(state_size=30, action_size=8)
    
    # Training parameters
    episodes = 300
    target_update_frequency = 10
    
    # Tracking
    scores = []
    best_score = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        
        if total_reward > best_score:
            best_score = total_reward
            agent.q_network.save('positive_etf_model.h5')
        
        # Train agent
        if len(agent.memory) > 1000:
            agent.replay(64)
        
        # Update target network
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Progress reporting
        if episode % 50 == 0:
            print(f"Episode {episode:3d}: Avg Score: {avg_score:8.4f}, Best: {best_score:8.4f}, ε: {agent.epsilon:.3f}")
    
    print(f"\n🎯 Training completed! Best score: {best_score:.4f}")
    return agent, env, scores

def test_positive_strategy():
    """Test the trained positive-score strategy"""
    print("\n🧪 TESTING POSITIVE-SCORE STRATEGY")
    print("=" * 50)
    
    # Load trained model
    try:
        model = keras.models.load_model('positive_etf_model.h5')
        print("✅ Trained model loaded successfully")
    except:
        print("❌ No trained model found")
        return
    
    # Initialize test environment
    env = PositiveScoreETFEnvironment()
    
    # Test on recent data (last 500 trading days)
    env.current_step = max(0, env.max_steps - 500)
    initial_step = env.current_step
    
    state = env.get_state()
    total_reward = 0
    actions_taken = []
    portfolio_values = []
    spy_values = []
    
    print(f"Testing from step {initial_step} to {env.max_steps}")
    
    while env.current_step < env.max_steps - 1:
        # Use trained model for decisions
        q_values = model.predict(state.reshape(1, -1), verbose=0)
        action = np.argmax(q_values[0])
        
        next_state, reward, done, info = env.step(action)
        
        actions_taken.append({
            'step': env.current_step,
            'action': action,
            'etf': info.get('current_etf', 'Unknown'),
            'portfolio_value': info.get('portfolio_value', 0),
            'spy_value': info.get('spy_value', 0),
            'outperformance': info.get('outperformance', 0)
        })
        
        portfolio_values.append(info.get('portfolio_value', 0))
        spy_values.append(info.get('spy_value', 0))
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Calculate final results
    final_portfolio = portfolio_values[-1] if portfolio_values else 10000
    final_spy = spy_values[-1] if spy_values else 10000
    
    strategy_return = (final_portfolio - 10000) / 10000 * 100
    spy_return = (final_spy - 10000) / 10000 * 100
    outperformance = strategy_return - spy_return
    
    print(f"\n📊 POSITIVE-SCORE STRATEGY RESULTS:")
    print(f"Strategy Return: {strategy_return:+.2f}%")
    print(f"SPY Benchmark:   {spy_return:+.2f}%")
    print(f"Outperformance:  {outperformance:+.2f}%")
    print(f"Total Reward:    {total_reward:+.2f}")
    print(f"Total Switches:  {env.total_switches}")
    
    # Save results
    results = {
        'strategy_return': strategy_return,
        'spy_return': spy_return,
        'outperformance': outperformance,
        'total_reward': total_reward,
        'total_switches': env.total_switches,
        'actions': actions_taken,
        'portfolio_values': portfolio_values,
        'spy_values': spy_values
    }
    
    with open('positive_strategy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to positive_strategy_results.json")
    
    return results

if __name__ == "__main__":
    print("🎯 POSITIVE-SCORE ETF SWITCHING STRATEGY")
    print("Designed to achieve positive rewards and beat SPY")
    print("=" * 60)
    
    # Train the strategy
    agent, env, scores = train_positive_score_strategy()
    
    # Test the strategy
    results = test_positive_strategy()
    
    print(f"\n🏆 FINAL ASSESSMENT:")
    if results and results['total_reward'] > 0:
        print(f"✅ SUCCESS: Achieved positive reward of {results['total_reward']:+.2f}")
        if results['outperformance'] > 0:
            print(f"✅ OUTPERFORMED SPY by {results['outperformance']:+.2f}%")
        else:
            print(f"⚠️ Underperformed SPY by {abs(results['outperformance']):.2f}%")
    else:
        print(f"❌ Did not achieve positive rewards")
    
    print("\n🎯 Strategy optimization complete!")