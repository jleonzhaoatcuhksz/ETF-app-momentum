"""
Run Backtesting Phase for Trained ETF Switching Model
"""

import numpy as np
import pandas as pd
import sqlite3
import json
from datetime import datetime
import tensorflow as tf
from ml_model import ETFSwitchingEnvironment, DQNAgent

def run_backtesting():
    """Run backtesting on the trained model"""
    
    print("=" * 60)
    print("🔬 RUNNING ETF SWITCHING BACKTEST")
    print("=" * 60)
    
    try:
        # Load environment
        print("📊 Loading environment and data...")
        env = ETFSwitchingEnvironment()
        
        # Create agent and load trained model
        print("🧠 Loading trained DQN model...")
        agent = DQNAgent(env.state_space_size, env.action_space_size)
        
        # Load the trained model with custom handling
        try:
            agent.q_network = tf.keras.models.load_model('etf_switching_model.h5', compile=False)
            # Recompile with correct loss function
            agent.q_network.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
        
        # Set epsilon to 0 for pure exploitation (no exploration)
        agent.epsilon = 0.0
        
        # Prepare test data (last 252 trading days = ~1 year)
        dates = sorted(env.data['date'].unique())
        test_dates = dates[-252:]  # Last year for testing
        
        print(f"📅 Testing period: {test_dates[0].date()} to {test_dates[-1].date()}")
        print(f"📈 Test days: {len(test_dates)}")
        
        # Initialize backtesting
        env.reset()
        state = env.get_state(test_dates[0])
        
        # Track results
        actions_taken = []
        portfolio_values = []
        total_reward = 0
        switches = 0
        
        print(f"🚀 Starting backtest...")
        
        for i, date in enumerate(test_dates):
            # Get model prediction
            if state is not None:
                action = agent.act(state)
                next_state, reward, done = env.step(action, date)
                
                # Record action
                action_record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'action': int(action),
                    'current_etf': env.current_etf,
                    'capital': float(env.current_capital),
                    'reward': float(reward)
                }
                actions_taken.append(action_record)
                portfolio_values.append(float(env.current_capital))
                
                # Count switches
                if action != 0:  # Non-hold action
                    switches += 1
                
                total_reward += reward
                state = next_state
                
                # Progress update
                if (i + 1) % 50 == 0:
                    print(f"   Day {i+1}/{len(test_dates)}: ${env.current_capital:,.2f}")
            
            else:
                break
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Initial Capital: ${initial_value:,.2f}")
        print(f"   Final Capital: ${final_value:,.2f}")
        print(f"   Total Return: {total_return*100:.2f}%")
        print(f"   Total Switches: {switches}")
        print(f"   Average Reward: {total_reward/len(test_dates):.4f}")
        
        # Save results
        results = {
            'actions': actions_taken,
            'portfolio_values': portfolio_values,
            'performance_metrics': {
                'initial_capital': initial_value,
                'final_capital': final_value,
                'total_return': total_return,
                'total_switches': switches,
                'total_reward': total_reward,
                'test_period': f"{test_dates[0].date()} to {test_dates[-1].date()}"
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open('etf_switching_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Results saved to 'etf_switching_results.json'")
        
        # Compare with SPY benchmark
        print(f"\n🏆 BENCHMARK COMPARISON:")
        compare_with_spy(test_dates, total_return)
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_spy(test_dates, strategy_return):
    """Compare strategy performance with SPY buy-and-hold"""
    
    try:
        conn = sqlite3.connect('./etf_data.db')
        
        # Get SPY prices for test period
        start_date = test_dates[0].strftime('%Y-%m-%d')
        end_date = test_dates[-1].strftime('%Y-%m-%d')
        
        query = """
        SELECT date, close 
        FROM prices 
        WHERE symbol = 'SPY' 
        AND date BETWEEN ? AND ?
        ORDER BY date
        """
        
        spy_data = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        if len(spy_data) > 1:
            spy_initial = spy_data['close'].iloc[0]
            spy_final = spy_data['close'].iloc[-1]
            spy_return = (spy_final - spy_initial) / spy_initial
            
            print(f"   SPY Return: {spy_return*100:.2f}%")
            print(f"   Strategy Return: {strategy_return*100:.2f}%")
            
            excess_return = strategy_return - spy_return
            print(f"   Excess Return: {excess_return*100:.2f}%")
            
            if excess_return > 0:
                print(f"   🎉 Strategy OUTPERFORMED SPY by {excess_return*100:.2f}%!")
            else:
                print(f"   📉 Strategy underperformed SPY by {abs(excess_return)*100:.2f}%")
        else:
            print(f"   ⚠️ Could not retrieve SPY benchmark data")
            
    except Exception as e:
        print(f"   ⚠️ Benchmark comparison failed: {e}")

if __name__ == "__main__":
    success = run_backtesting()
    if success:
        print(f"\n🎯 BACKTEST COMPLETED SUCCESSFULLY!")
        print(f"   Check 'etf_switching_results.json' for detailed results")
    else:
        print(f"\n❌ BACKTEST FAILED - Check error messages above")