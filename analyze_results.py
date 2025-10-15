"""
Analysis and Visualization Script for ETF Switching Strategy Results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3

def load_results(filename='etf_switching_results.json'):
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        print(f"Results file {filename} not found. Please run the ML model first.")
        return None

def analyze_performance(results):
    """Analyze strategy performance"""
    if not results:
        return
    
    actions_df = pd.DataFrame(results['actions'])
    actions_df['date'] = pd.to_datetime(actions_df['date'])
    
    portfolio_values = results['portfolio_values']
    
    print("ETF Switching Strategy Analysis")
    print("=" * 40)
    
    # Basic performance metrics
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    
    print(f"Initial Capital: ${initial_value:,.2f}")
    print(f"Final Capital: ${final_value:,.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    
    # Calculate daily returns
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return)
    
    daily_returns = np.array(daily_returns)
    
    # Risk metrics
    volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
    sharpe_ratio = (total_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
    
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    print(f"Annualized Volatility: {volatility*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    
    # Switching analysis
    switches = actions_df[actions_df['action'] != 0]  # Non-hold actions
    total_switches = len(switches)
    
    print(f"\nSwitching Activity:")
    print(f"Total Switches: {total_switches}")
    print(f"Average Days Between Switches: {len(actions_df) / max(total_switches, 1):.1f}")
    
    # ETF preferences
    etf_counts = actions_df['current_etf'].value_counts()
    print(f"\nETF Holdings (Top 5):")
    for etf, count in etf_counts.head().items():
        percentage = (count / len(actions_df)) * 100
        print(f"  {etf}: {percentage:.1f}% of time")
    
    return actions_df, portfolio_values, {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_switches': total_switches
    }

def plot_performance(actions_df, portfolio_values):
    """Create performance visualizations"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ETF Switching Strategy Performance Analysis', fontsize=16)
    
    # 1. Portfolio Value Over Time
    axes[0, 0].plot(actions_df['date'], portfolio_values, linewidth=2, color='blue')
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Daily Returns Distribution
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        daily_returns.append(daily_return * 100)  # Convert to percentage
    
    axes[0, 1].hist(daily_returns, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].set_xlabel('Daily Return (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ETF Holdings Over Time
    etf_changes = actions_df[actions_df['current_etf'] != actions_df['current_etf'].shift()]
    
    # Create a timeline of ETF holdings
    unique_etfs = actions_df['current_etf'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_etfs)))
    
    current_etf = actions_df['current_etf'].iloc[0]
    start_date = actions_df['date'].iloc[0]
    
    for i, (_, row) in enumerate(etf_changes.iterrows()):
        if i > 0:  # Skip first row
            # Plot the previous ETF period
            end_date = row['date']
            color = colors[list(unique_etfs).index(current_etf)]
            axes[1, 0].barh(0, (end_date - start_date).days, left=(start_date - actions_df['date'].iloc[0]).days, 
                           height=0.5, color=color, alpha=0.7, label=current_etf if current_etf not in axes[1, 0].get_legend_handles_labels()[1] else "")
            
            start_date = end_date
            current_etf = row['current_etf']
    
    # Plot the final period
    end_date = actions_df['date'].iloc[-1]
    color = colors[list(unique_etfs).index(current_etf)]
    axes[1, 0].barh(0, (end_date - start_date).days, left=(start_date - actions_df['date'].iloc[0]).days, 
                   height=0.5, color=color, alpha=0.7)
    
    axes[1, 0].set_title('ETF Holdings Timeline')
    axes[1, 0].set_xlabel('Days')
    axes[1, 0].set_ylabel('')
    axes[1, 0].set_yticks([])
    
    # 4. Monthly Trend vs Action Rewards
    monthly_rewards = actions_df.groupby(actions_df['date'].dt.to_period('M'))['reward'].sum()
    
    axes[1, 1].plot(monthly_rewards.index.astype(str), monthly_rewards.values, marker='o', linewidth=2)
    axes[1, 1].set_title('Monthly Rewards')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('etf_switching_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_benchmark():
    """Compare strategy performance with buy-and-hold SPY"""
    
    # Load SPY data for benchmark
    conn = sqlite3.connect('./etf_data.db')
    spy_query = """
    SELECT date, close 
    FROM prices 
    WHERE symbol = 'SPY' AND monthly_trend IS NOT NULL
    ORDER BY date
    """
    
    spy_df = pd.read_sql_query(spy_query, conn)
    conn.close()
    
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    
    # Calculate SPY buy-and-hold return
    initial_spy_price = spy_df['close'].iloc[0]
    final_spy_price = spy_df['close'].iloc[-1]
    spy_return = (final_spy_price - initial_spy_price) / initial_spy_price
    
    print(f"\nBenchmark Comparison (SPY Buy & Hold):")
    print(f"SPY Total Return: {spy_return*100:.2f}%")
    
    return spy_return

def generate_report():
    """Generate comprehensive analysis report"""
    
    results = load_results()
    if not results:
        return
    
    print("Generating comprehensive analysis report...")
    
    # Perform analysis
    actions_df, portfolio_values, metrics = analyze_performance(results)
    
    # Create visualizations
    plot_performance(actions_df, portfolio_values)
    
    # Compare with benchmark
    spy_return = compare_with_benchmark()
    
    # Generate summary report
    strategy_return = metrics['total_return']
    excess_return = strategy_return - spy_return
    
    print(f"\n" + "="*50)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Strategy Return:     {strategy_return*100:.2f}%")
    print(f"Benchmark (SPY):     {spy_return*100:.2f}%")
    print(f"Excess Return:       {excess_return*100:.2f}%")
    print(f"Volatility:          {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:        {metrics['max_drawdown']*100:.2f}%")
    print(f"Total Switches:      {metrics['total_switches']}")
    
    if excess_return > 0:
        print(f"\n✅ Strategy OUTPERFORMED benchmark by {excess_return*100:.2f}%")
    else:
        print(f"\n❌ Strategy UNDERPERFORMED benchmark by {abs(excess_return)*100:.2f}%")
    
    print("\nAnalysis complete! Charts saved as 'etf_switching_analysis.png'")

if __name__ == "__main__":
    generate_report()