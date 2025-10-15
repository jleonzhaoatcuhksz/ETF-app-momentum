"""
Improved ETF Switching Strategy - Designed to Beat SPY
Key Improvements:
1. Weekly decisions (not daily) to reduce overtrading
2. Momentum-based switching with trend confirmation
3. Higher transaction cost penalty
4. Risk-adjusted reward system
5. Minimum holding periods
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import json
from collections import deque
import random

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

class ImprovedETFEnvironment:
    """
    Improved ETF Environment focused on beating SPY
    """
    
    def __init__(self, db_path='./etf_data.db', initial_capital=10000, transaction_cost=0.005):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.5% to discourage overtrading
        
        # Load and process data
        self.data = self._load_data()
        self.etf_symbols = self.data['symbol'].unique().tolist()
        self.n_etfs = len(self.etf_symbols)
        
        # State variables
        self.current_etf = 'SPY'  # Start with SPY
        self.current_position = 0
        self.days_held = 0
        self.min_holding_days = 5  # Minimum 5 days holding period
        self.current_day_idx = 0
        
        # Weekly decision making (every 5 days)
        self.decision_frequency = 5
        
        # Action space: Hold + Switch to top performing ETFs only
        self.action_space_size = 8  # Hold + top 7 ETFs
        
        # Enhanced state space with momentum and risk features
        self.state_space_size = 25  # More comprehensive features
        
        print(f"Improved Environment initialized:")
        print(f"  - ETFs: {self.n_etfs} symbols")
        print(f"  - Decision frequency: Every {self.decision_frequency} days")
        print(f"  - Transaction cost: {self.transaction_cost*100:.1f}%")
        print(f"  - Min holding period: {self.min_holding_days} days")

    def _load_data(self):
        """Load and enhance data with momentum indicators"""
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
        
        # Calculate additional features for each ETF
        enhanced_data = []
        
        for symbol in df['symbol'].unique():
            etf_data = df[df['symbol'] == symbol].copy()
            etf_data = etf_data.sort_values('date')
            
            # Calculate momentum indicators
            etf_data['return_1d'] = etf_data['close'].pct_change()
            etf_data['return_5d'] = etf_data['close'].pct_change(5)
            etf_data['return_20d'] = etf_data['close'].pct_change(20)
            
            # Volatility (20-day rolling)
            etf_data['volatility'] = etf_data['return_1d'].rolling(20).std() * np.sqrt(252)
            
            # Momentum score (combination of trends)
            etf_data['momentum_score'] = (
                etf_data['return_5d'] * 0.3 + 
                etf_data['return_20d'] * 0.4 + 
                etf_data['monthly_trend'] * 0.01 * 0.3
            )
            
            # Risk-adjusted momentum (Sharpe-like)
            etf_data['risk_adj_momentum'] = etf_data['momentum_score'] / (etf_data['volatility'] + 0.01)
            
            enhanced_data.append(etf_data)
        
        enhanced_df = pd.concat(enhanced_data, ignore_index=True)
        enhanced_df = enhanced_df.fillna(0)
        
        print(f"Enhanced data loaded: {len(enhanced_df)} records with momentum indicators")
        return enhanced_df

    def get_state(self, date):
        """Get enhanced state with momentum and risk features"""
        day_data = self.data[self.data['date'] == date]
        
        if len(day_data) == 0:
            return None
        
        # Current ETF features (5 dimensions)
        current_etf_data = day_data[day_data['symbol'] == self.current_etf]
        if len(current_etf_data) > 0:
            current_features = [
                current_etf_data['monthly_trend'].iloc[0] / 100,
                current_etf_data['return_5d'].iloc[0],
                current_etf_data['return_20d'].iloc[0],
                current_etf_data['volatility'].iloc[0] / 100,
                current_etf_data['risk_adj_momentum'].iloc[0]
            ]
        else:
            current_features = [0] * 5
        
        # Top ETFs momentum ranking (7 dimensions)
        day_data_sorted = day_data.sort_values('risk_adj_momentum', ascending=False)
        top_etfs_momentum = day_data_sorted['risk_adj_momentum'].head(7).tolist()
        while len(top_etfs_momentum) < 7:
            top_etfs_momentum.append(0)
        
        # Market regime indicators (5 dimensions)
        spy_data = day_data[day_data['symbol'] == 'SPY']
        if len(spy_data) > 0:
            spy_trend = spy_data['monthly_trend'].iloc[0] / 100
            spy_vol = spy_data['volatility'].iloc[0] / 100
            spy_momentum = spy_data['risk_adj_momentum'].iloc[0]
        else:
            spy_trend, spy_vol, spy_momentum = 0, 0, 0
        
        market_features = [
            spy_trend,
            spy_vol,
            spy_momentum,
            1 if spy_trend > 0 else 0,  # Bull/bear market
            1 if spy_vol < 0.2 else 0   # Low volatility regime
        ]
        
        # Portfolio state (8 dimensions)
        portfolio_performance = (self.current_capital - self.initial_capital) / self.initial_capital
        portfolio_features = [
            self.days_held / 30.0,                    # Days held (normalized)
            1 if self.days_held >= self.min_holding_days else 0,  # Can switch
            portfolio_performance,                     # Total return
            max(-0.5, min(0.5, portfolio_performance)), # Clipped return
            self.etf_symbols.index(self.current_etf) / len(self.etf_symbols), # Current ETF ID
            1 if self.current_etf == 'SPY' else 0,    # Holding SPY
            1 if self.current_etf in ['XLK', 'XLV', 'XLF'] else 0, # Growth sectors
            1 if self.current_etf in ['XLE', 'XLB', 'XLI'] else 0  # Value sectors
        ]
        
        # Combine all features (5 + 7 + 5 + 8 = 25)
        state = current_features + top_etfs_momentum + market_features + portfolio_features
        return np.array(state, dtype=np.float32)

    def step(self, action, date):
        """Execute action with improved reward system"""
        day_data = self.data[self.data['date'] == date]
        
        if len(day_data) == 0:
            return None, 0, True
        
        # Get current ETF price
        current_etf_data = day_data[day_data['symbol'] == self.current_etf]
        if len(current_etf_data) == 0:
            return None, 0, True
        
        current_price = current_etf_data['close'].iloc[0]
        reward = 0
        
        # Increment days held
        self.days_held += 1
        
        # Action 0: Hold current position
        if action == 0:
            # Reward for holding (avoid transaction costs)
            if self.current_position > 0:
                current_value = self.current_position * current_price
                daily_return = (current_value - self.current_capital) / self.current_capital
                reward = daily_return
                self.current_capital = current_value
        
        # Actions 1-7: Switch to top performing ETFs
        else:
            # Only allow switching if minimum holding period is met
            if self.days_held < self.min_holding_days:
                reward = -0.02  # Penalty for trying to switch too early
            else:
                # Get top performing ETFs
                day_data_sorted = day_data.sort_values('risk_adj_momentum', ascending=False)
                top_etfs = day_data_sorted['symbol'].head(7).tolist()
                
                if action <= len(top_etfs):
                    target_etf = top_etfs[action - 1]
                    target_data = day_data[day_data['symbol'] == target_etf]
                    
                    if len(target_data) > 0 and target_etf != self.current_etf:
                        target_price = target_data['close'].iloc[0]
                        
                        # Sell current position
                        if self.current_position > 0:
                            sell_value = self.current_position * current_price
                            cash_after_sell = sell_value * (1 - self.transaction_cost)
                        else:
                            cash_after_sell = self.current_capital
                        
                        # Buy new position
                        self.current_position = cash_after_sell / target_price
                        self.current_capital = self.current_position * target_price
                        
                        # Calculate reward based on expected performance
                        target_momentum = target_data['risk_adj_momentum'].iloc[0]
                        target_trend = target_data['monthly_trend'].iloc[0] / 100
                        
                        # Reward = expected return - transaction costs
                        expected_return = target_momentum * 0.1 + target_trend * 0.05
                        reward = expected_return - (self.transaction_cost * 2)
                        
                        # Update state
                        self.current_etf = target_etf
                        self.days_held = 0  # Reset holding period
                        
                    else:
                        reward = -0.01  # Small penalty for invalid switch
        
        # Get next state
        next_state = self.get_state(date)
        
        # Check if episode is done
        done = self.current_day_idx >= len(self.data['date'].unique()) - 1
        self.current_day_idx += 1
        
        return next_state, reward, done

    def reset(self):
        """Reset environment"""
        self.current_capital = self.initial_capital
        self.current_etf = 'SPY'
        self.days_held = 0
        self.current_day_idx = 0
        
        # Get initial SPY price
        first_date = self.data['date'].min()
        spy_data = self.data[(self.data['date'] == first_date) & (self.data['symbol'] == 'SPY')]
        if len(spy_data) > 0:
            spy_price = spy_data['close'].iloc[0]
            self.current_position = self.initial_capital / spy_price
            self.current_capital = self.current_position * spy_price
        
        return self.get_state(first_date)

class ImprovedDQNAgent:
    """Improved DQN with better architecture"""
    
    def __init__(self, state_size, action_size, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.998  # Slower decay
        self.epsilon_min = 0.05     # Higher minimum for exploration
        self.memory = deque(maxlen=20000)  # Larger memory
        self.batch_size = 64
        
        if HAS_TF:
            self.q_network = self._build_model()
            self.target_network = self._build_model()
            self.update_target_network()

    def _build_model(self):
        """Build improved neural network"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber'  # More robust loss function
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not HAS_TF:
            return random.randint(0, self.action_size - 1)
            
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if not HAS_TF or len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch if e[3] is not None])
        dones = np.array([e[4] for e in batch])
        
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.99 * np.max(next_q_values[i])
        
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        if HAS_TF:
            self.target_network.set_weights(self.q_network.get_weights())

def train_improved_strategy(episodes=300):
    """Train the improved strategy"""
    
    print("🚀 TRAINING IMPROVED ETF SWITCHING STRATEGY")
    print("=" * 60)
    
    env = ImprovedETFEnvironment()
    agent = ImprovedDQNAgent(env.state_space_size, env.action_space_size)
    
    scores = deque(maxlen=100)
    best_score = -np.inf
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Use weekly decision points (every 5 days)
        dates = sorted(env.data['date'].unique())
        decision_dates = dates[::env.decision_frequency][:200]  # ~4 years per episode
        
        for date in decision_dates:
            if state is not None:
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
        
        # Update target network
        if episode % 50 == 0:
            agent.update_target_network()
        
        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            if HAS_TF:
                agent.q_network.save('improved_etf_model.h5')
        
        # Progress report
        if episode % 50 == 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode:3d}: Avg Score: {avg_score:8.4f}, Best: {best_score:8.4f}, ε: {agent.epsilon:.3f}")
    
    print(f"✅ Training completed! Best score: {best_score:.4f}")
    return agent, env

def test_improved_strategy():
    """Test the improved strategy"""
    
    print("\n🧪 TESTING IMPROVED STRATEGY")
    print("=" * 60)
    
    try:
        env = ImprovedETFEnvironment()
        agent = ImprovedDQNAgent(env.state_space_size, env.action_space_size)
        
        # Load best model
        if HAS_TF:
            agent.q_network = tf.keras.models.load_model('improved_etf_model.h5', compile=False)
            agent.q_network.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss='huber'
            )
            agent.epsilon = 0.0  # No exploration during testing
        
        # Test on last 2 years
        dates = sorted(env.data['date'].unique())
        test_dates = dates[-500::5]  # Weekly decisions for last 2 years
        
        env.reset()
        state = env.get_state(test_dates[0])
        
        results = []
        portfolio_values = []
        switches = 0
        
        print(f"📅 Test period: {test_dates[0].date()} to {test_dates[-1].date()}")
        
        for i, date in enumerate(test_dates):
            if state is not None:
                action = agent.act(state)
                next_state, reward, done = env.step(action, date)
                
                results.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'action': int(action),
                    'etf': env.current_etf,
                    'capital': float(env.current_capital),
                    'days_held': env.days_held
                })
                
                portfolio_values.append(float(env.current_capital))
                
                if action != 0:
                    switches += 1
                
                state = next_state
                
                if (i + 1) % 20 == 0:
                    print(f"   Week {i+1:3d}: ${env.current_capital:,.2f} ({env.current_etf})")
        
        # Calculate performance
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        strategy_return = (final_value - initial_value) / initial_value
        
        # Compare with SPY
        spy_initial = get_spy_price(test_dates[0])
        spy_final = get_spy_price(test_dates[-1])
        spy_return = (spy_final - spy_initial) / spy_initial
        
        print(f"\n📊 IMPROVED STRATEGY RESULTS:")
        print(f"   Strategy Return: {strategy_return*100:+.2f}%")
        print(f"   SPY Return:      {spy_return*100:+.2f}%")
        print(f"   Excess Return:   {(strategy_return-spy_return)*100:+.2f}%")
        print(f"   Total Switches:  {switches}")
        print(f"   Avg Hold Period: {len(test_dates)/max(switches,1):.1f} weeks")
        
        if strategy_return > spy_return:
            print(f"   🎉 SUCCESS! Strategy BEAT SPY by {(strategy_return-spy_return)*100:.2f}%")
        else:
            print(f"   📉 Strategy underperformed SPY by {(spy_return-strategy_return)*100:.2f}%")
        
        # Save detailed results
        detailed_results = {
            'strategy_return': strategy_return,
            'spy_return': spy_return,
            'excess_return': strategy_return - spy_return,
            'total_switches': switches,
            'trading_log': results,
            'portfolio_values': portfolio_values
        }
        
        with open('improved_strategy_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"✅ Detailed results saved to 'improved_strategy_results.json'")
        
        return strategy_return > spy_return
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def get_spy_price(date):
    """Get SPY price for a specific date"""
    conn = sqlite3.connect('./etf_data.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT close FROM prices WHERE symbol = 'SPY' AND date = ? LIMIT 1",
        (date.strftime('%Y-%m-%d'),)
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 100

if __name__ == "__main__":
    print("🎯 IMPROVED ETF SWITCHING STRATEGY")
    print("Designed to beat SPY buy-and-hold")
    print("=" * 60)
    
    if not HAS_TF:
        print("❌ TensorFlow required. Install with: pip install tensorflow")
        exit(1)
    
    # Train improved strategy
    agent, env = train_improved_strategy(episodes=300)
    
    # Test the strategy
    success = test_improved_strategy()
    
    if success:
        print(f"\n🏆 MISSION ACCOMPLISHED!")
        print(f"The improved strategy successfully beat SPY!")
    else:
        print(f"\n🔄 Need further improvements...")
        print(f"Consider adjusting parameters or trying different approaches.")