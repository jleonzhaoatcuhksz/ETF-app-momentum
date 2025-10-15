"""
ML-Based ETF Switching Strategy using Deep Q-Network (DQN)
Single ETF with Dynamic Switching based on Monthly_Trend analysis
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from collections import deque
import random

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    HAS_TF = False

class ETFSwitchingEnvironment:
    """
    Reinforcement Learning Environment for ETF Switching Strategy
    """
    
    def __init__(self, db_path='./etf_data.db', initial_capital=10000, transaction_cost=0.001):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% transaction cost
        
        # Load data
        self.data = self._load_data()
        self.etf_symbols = self.data['symbol'].unique().tolist()
        self.n_etfs = len(self.etf_symbols)
        
        # State variables
        self.current_etf = 'SPY'  # Start with SPY
        self.current_position = 0  # Number of shares
        self.current_day_idx = 0
        self.total_days = len(self.data['date'].unique())
        
        # Action space: Hold + Switch to any of the 14 ETFs
        self.action_space_size = self.n_etfs + 1  # Hold + 14 ETFs
        
        # State space: [current_etf_id, monthly_trends_all_etfs, days_held, portfolio_performance]
        self.state_space_size = 1 + self.n_etfs + 2  # current_etf + trends + days_held + performance
        
        print(f"Environment initialized:")
        print(f"  - ETFs: {self.n_etfs} symbols")
        print(f"  - Data range: {self.total_days} days")
        print(f"  - Action space: {self.action_space_size}")
        print(f"  - State space: {self.state_space_size}")

    def _load_data(self):
        """Load and preprocess data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT symbol, date, close, sma_5d, monthly_trend
        FROM prices 
        WHERE monthly_trend IS NOT NULL 
        ORDER BY date, symbol
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Fill missing values
        df['monthly_trend'] = df['monthly_trend'].fillna(0)
        
        print(f"Data loaded: {len(df)} records")
        return df

    def get_state(self, date):
        """Get current state for the given date"""
        # Get data for all ETFs on this date
        day_data = self.data[self.data['date'] == date]
        
        if len(day_data) == 0:
            return None
            
        # Current ETF ID (one-hot encoded)
        current_etf_id = self.etf_symbols.index(self.current_etf)
        
        # Monthly trends for all ETFs
        trends = []
        for symbol in self.etf_symbols:
            etf_data = day_data[day_data['symbol'] == symbol]
            if len(etf_data) > 0:
                trends.append(etf_data['monthly_trend'].iloc[0])
            else:
                trends.append(0)  # Missing data
        
        # Days held current ETF
        days_held = min(self.current_day_idx, 30)  # Cap at 30 days
        
        # Portfolio performance (normalized)
        performance = (self.current_capital - self.initial_capital) / self.initial_capital
        performance = max(-1, min(1, performance))  # Clip to [-1, 1]
        
        state = [current_etf_id] + trends + [days_held / 30.0, performance]
        return np.array(state, dtype=np.float32)

    def step(self, action, date):
        """Execute action and return new state, reward, done"""
        day_data = self.data[self.data['date'] == date]
        
        if len(day_data) == 0:
            return None, 0, True  # End episode if no data
        
        # Current ETF price
        current_etf_data = day_data[day_data['symbol'] == self.current_etf]
        if len(current_etf_data) == 0:
            return None, 0, True
            
        current_price = current_etf_data['close'].iloc[0]
        
        reward = 0
        
        # Action 0: Hold current position
        if action == 0:
            # No transaction, just track performance
            if self.current_position > 0:
                current_value = self.current_position * current_price
                reward = (current_value - self.current_capital) / self.current_capital
        
        # Actions 1-14: Switch to specific ETF
        else:
            target_etf = self.etf_symbols[action - 1]
            target_data = day_data[day_data['symbol'] == target_etf]
            
            if len(target_data) == 0:
                reward = -0.01  # Penalty for invalid action
            else:
                target_price = target_data['close'].iloc[0]
                
                # Only switch if different from current ETF
                if target_etf != self.current_etf:
                    # Sell current position
                    if self.current_position > 0:
                        sell_value = self.current_position * current_price
                        self.current_capital = sell_value * (1 - self.transaction_cost)
                    
                    # Buy new position
                    self.current_position = self.current_capital / target_price
                    self.current_capital = self.current_position * target_price
                    self.current_etf = target_etf
                    
                    # Reward based on monthly trend of target ETF
                    target_trend = target_data['monthly_trend'].iloc[0]
                    reward = target_trend / 100.0  # Convert percentage to decimal
                    
                    # Apply transaction cost penalty
                    reward -= self.transaction_cost * 2  # Buy + Sell costs
                else:
                    # No switch needed, small penalty for redundant action
                    reward = -0.001
        
        # Get next state
        next_state = self.get_state(date)
        
        # Check if episode is done
        done = self.current_day_idx >= self.total_days - 1
        
        # Update day index
        self.current_day_idx += 1
        
        return next_state, reward, done

    def reset(self):
        """Reset environment to initial state"""
        self.current_capital = self.initial_capital
        self.current_etf = 'SPY'
        self.current_position = self.initial_capital / self._get_price('SPY', 0)
        self.current_day_idx = 0
        
        first_date = self.data['date'].unique()[0]
        return self.get_state(first_date)

    def _get_price(self, symbol, day_idx):
        """Get price for symbol on specific day"""
        dates = sorted(self.data['date'].unique())
        if day_idx >= len(dates):
            return 100  # Default price
            
        date = dates[day_idx]
        day_data = self.data[(self.data['date'] == date) & (self.data['symbol'] == symbol)]
        
        if len(day_data) > 0:
            return day_data['close'].iloc[0]
        return 100  # Default price


class DQNAgent:
    """Deep Q-Network Agent for ETF Switching"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        if HAS_TF:
            self.q_network = self._build_model()
            self.target_network = self._build_model()
            self.update_target_network()
        else:
            print("TensorFlow not available. DQN agent disabled.")

    def _build_model(self):
        """Build neural network model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if not HAS_TF:
            return random.randint(0, self.action_size - 1)
            
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model on a batch of experiences"""
        if not HAS_TF or len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch if e[3] is not None])
        dones = np.array([e[4] for e in batch])
        
        # Predict Q-values for current states
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Predict Q-values for next states using target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # Train the model
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update target network with current network weights"""
        if HAS_TF:
            self.target_network.set_weights(self.q_network.get_weights())


def train_etf_switching_model(episodes=1000):
    """Train the ETF switching model"""
    
    # Initialize environment and agent
    env = ETFSwitchingEnvironment()
    agent = DQNAgent(env.state_space_size, env.action_space_size)
    
    scores = deque(maxlen=100)
    
    print(f"\nStarting training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Get all dates for this episode
        dates = sorted(env.data['date'].unique())
        
        for date in dates[:min(252, len(dates))]:  # Limit to ~1 year per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action, date)
            
            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
        
        scores.append(total_reward)
        
        # Train the agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.4f}, Epsilon: {agent.epsilon:.3f}")
    
    print("Training completed!")
    
    # Save the model
    if HAS_TF:
        agent.q_network.save('etf_switching_model.h5')
        print("Model saved as 'etf_switching_model.h5'")
    
    return agent, env


def test_strategy(agent, env, test_days=252):
    """Test the trained strategy"""
    state = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = []
    
    dates = sorted(env.data['date'].unique())[-test_days:]  # Last year for testing
    
    print(f"\nTesting strategy on {len(dates)} days...")
    
    for i, date in enumerate(dates):
        action = agent.act(state)
        next_state, reward, done = env.step(action, date)
        
        actions_taken.append({
            'date': date.strftime('%Y-%m-%d'),
            'action': action,
            'current_etf': env.current_etf,
            'capital': env.current_capital,
            'reward': reward
        })
        
        portfolio_values.append(env.current_capital)
        total_reward += reward
        
        if next_state is not None:
            state = next_state
        
        if done:
            break
    
    # Calculate performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    
    print(f"\nStrategy Performance:")
    print(f"Initial Capital: ${initial_value:,.2f}")
    print(f"Final Capital: ${final_value:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Total Reward: {total_reward:.4f}")
    
    return actions_taken, portfolio_values


if __name__ == "__main__":
    print("ETF Switching Strategy with Deep Q-Network")
    print("=" * 50)
    
    # Check if we have the required dependencies
    if not HAS_TF:
        print("Please install TensorFlow: pip install tensorflow")
        print("Running in demo mode with random actions...")
    
    # Train the model
    try:
        agent, env = train_etf_switching_model(episodes=500)
        
        # Test the strategy
        actions, values = test_strategy(agent, env)
        
        # Save results
        results = {
            'actions': actions,
            'portfolio_values': values,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('etf_switching_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nResults saved to 'etf_switching_results.json'")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Please check your database connection and data availability.")