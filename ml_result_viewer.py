"""
Comprehensive ML Result Viewer
Training, Validation, Testing Results with Trading Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import os
from pathlib import Path

class MLResultViewer:
    def __init__(self):
        """Initialize ML Result Viewer"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("📊 ML RESULT VIEWER")
        print("Comprehensive analysis of training, validation & testing")
        print("=" * 60)
        
        self.results = {}
        self.load_all_results()
        
    def load_all_results(self):
        """Load all available ML results"""
        print("📂 Loading ML results...")
        
        # Result files to check
        result_files = [
            'best_fast_ml_results.json',
            'random_forest_results.json', 
            'xgboost_results.json',
            'lightgbm_results.json',
            'positive_strategy_results.json',
            'improved_strategy_results.json',
            'etf_switching_results.json'
        ]
        
        for file in result_files:
            if os.path.exists(file):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    model_name = file.replace('_results.json', '').replace('.json', '')
                    self.results[model_name] = data
                    print(f"✅ Loaded: {file}")
                    
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
            else:
                print(f"⚠️ Not found: {file}")
        
        print(f"📊 Total results loaded: {len(self.results)}")
        
    def display_training_summary(self):
        """Display comprehensive training summary"""
        print("\n" + "="*60)
        print("🤖 TRAINING PHASE SUMMARY")
        print("="*60)
        
        # Training performance comparison
        training_data = []
        
        for model_name, results in self.results.items():
            if 'model_name' in results:
                training_data.append({
                    'Model': results['model_name'].upper(),
                    'Strategy Return (%)': f"{results.get('strategy_return', 0):.2f}",
                    'SPY Return (%)': f"{results.get('spy_return', 0):.2f}",
                    'Outperformance (%)': f"{results.get('outperformance', 0):.2f}",
                    'Total Trades': results.get('total_trades', 0),
                    'Status': '🏆 SUCCESS' if results.get('outperformance', 0) > 0 else '⚠️ UNDERPERFORMED'
                })
        
        if training_data:
            df = pd.DataFrame(training_data)
            print(df.to_string(index=False))
        else:
            print("No detailed training results available")
            
        # Best performing model
        if training_data:
            best_model = max(training_data, key=lambda x: float(x['Outperformance (%)']))
            print(f"\n🥇 BEST PERFORMING MODEL: {best_model['Model']}")
            print(f"   Outperformance: {best_model['Outperformance (%)']}%")
            print(f"   Total Trades: {best_model['Total Trades']}")
    
    def display_validation_analysis(self):
        """Display validation phase analysis"""
        print("\n" + "="*60)
        print("✅ VALIDATION PHASE ANALYSIS")
        print("="*60)
        
        # Load database for validation analysis
        try:
            conn = sqlite3.connect('./etf_data.db')
            
            # Get data statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices")
            total_etfs = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
            date_range = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*) FROM prices WHERE monthly_trend IS NOT NULL")
            valid_records = cursor.fetchone()[0]
            
            print(f"📊 DATA VALIDATION:")
            print(f"   Total Records: {total_records:,}")
            print(f"   Valid Records: {valid_records:,} ({valid_records/total_records*100:.1f}%)")
            print(f"   ETFs Available: {total_etfs}")
            print(f"   Date Range: {date_range[0]} to {date_range[1]}")
            
            # Feature validation
            cursor.execute("""
                SELECT symbol, COUNT(*) as records, 
                       AVG(monthly_trend) as avg_trend,
                       MIN(monthly_trend) as min_trend,
                       MAX(monthly_trend) as max_trend
                FROM prices 
                WHERE monthly_trend IS NOT NULL 
                GROUP BY symbol 
                ORDER BY records DESC
            """)
            
            etf_stats = cursor.fetchall()
            
            print(f"\n📈 ETF FEATURE VALIDATION:")
            print(f"{'ETF':<6} {'Records':<8} {'Avg Trend':<10} {'Min Trend':<10} {'Max Trend'}")
            print("-" * 50)
            
            for etf, records, avg_trend, min_trend, max_trend in etf_stats[:10]:
                print(f"{etf:<6} {records:<8,} {avg_trend:<10.2f} {min_trend:<10.2f} {max_trend:.2f}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
    
    def display_detailed_testing_results(self):
        """Display comprehensive testing results with step-by-step analysis"""
        print("\n" + "="*60)
        print("🧪 DETAILED TESTING RESULTS")
        print("="*60)
        
        for model_name, results in self.results.items():
            if 'trades' in results and results['trades']:
                self.analyze_model_testing(model_name, results)
    
    def analyze_model_testing(self, model_name, results):
        """Analyze individual model testing results"""
        print(f"\n🔍 {model_name.upper()} TESTING ANALYSIS")
        print("-" * 40)
        
        # Overall performance
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Strategy Return: {results.get('strategy_return', 0):+.2f}%")
        print(f"   SPY Benchmark:   {results.get('spy_return', 0):+.2f}%")
        print(f"   Outperformance:  {results.get('outperformance', 0):+.2f}%")
        print(f"   Final Portfolio: ${results.get('final_portfolio_value', 0):,.2f}")
        print(f"   Final SPY Value: ${results.get('final_spy_value', 0):,.2f}")
        
        # Trading analysis
        trades = results.get('trades', [])
        if trades:
            print(f"\n💼 TRADING ANALYSIS:")
            print(f"   Total Trades: {len(trades)}")
            
            # Trade details
            print(f"\n📋 TRADE-BY-TRADE ANALYSIS:")
            print(f"{'Date':<12} {'From':<5} {'To':<5} {'Confidence':<10} {'Portfolio Value':<15} {'P&L'}")
            print("-" * 70)
            
            previous_value = 10000  # Starting value
            
            for i, trade in enumerate(trades):
                try:
                    current_value = float(trade.get('portfolio_value', 0))
                    pnl = current_value - previous_value
                    pnl_pct = (pnl / previous_value) * 100 if previous_value > 0 else 0
                    
                    date_str = str(trade.get('date', ''))[:10]  # First 10 chars of date
                    from_etf = str(trade.get('from_etf', 'N/A'))[:5]
                    to_etf = str(trade.get('to_etf', 'N/A'))[:5]
                    confidence = float(trade.get('confidence', 0))
                    
                    pnl_symbol = "📈" if pnl >= 0 else "📉"
                    
                    print(f"{date_str:<12} {from_etf:<5} {to_etf:<5} {confidence:<10.3f} ${current_value:<13,.2f} {pnl_symbol} {pnl:+.2f} ({pnl_pct:+.1f}%)")
                    
                    previous_value = current_value
                    
                except (ValueError, TypeError) as e:
                    print(f"Trade {i+1}: Data formatting error - {e}")
                    continue
            
            # Trading statistics
            portfolio_values = [trade.get('portfolio_value', 0) for trade in trades]
            if len(portfolio_values) > 1:
                trade_returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100 
                               for i in range(1, len(portfolio_values))]
                
                winning_trades = sum(1 for ret in trade_returns if ret > 0)
                losing_trades = sum(1 for ret in trade_returns if ret < 0)
                
                print(f"\n📊 TRADING STATISTICS:")
                print(f"   Winning Trades: {winning_trades} ({winning_trades/len(trade_returns)*100:.1f}%)")
                print(f"   Losing Trades:  {losing_trades} ({losing_trades/len(trade_returns)*100:.1f}%)")
                print(f"   Average Trade:  {np.mean(trade_returns):+.2f}%")
                print(f"   Best Trade:     {max(trade_returns):+.2f}%")
                print(f"   Worst Trade:    {min(trade_returns):+.2f}%")
        
        # ETF allocation analysis
        if trades:
            etf_trades = {}
            for trade in trades:
                to_etf = trade.get('to_etf', 'Unknown')
                etf_trades[to_etf] = etf_trades.get(to_etf, 0) + 1
            
            print(f"\n🎯 ETF ALLOCATION ANALYSIS:")
            for etf, count in sorted(etf_trades.items(), key=lambda x: x[1], reverse=True):
                print(f"   {etf}: {count} trades ({count/len(trades)*100:.1f}%)")
    
    def create_performance_charts(self):
        """Create comprehensive performance charts"""
        print(f"\n📈 CREATING PERFORMANCE CHARTS...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Model Comparison Chart
        ax1 = plt.subplot(2, 3, 1)
        self.plot_model_comparison(ax1)
        
        # 2. Returns Distribution
        ax2 = plt.subplot(2, 3, 2) 
        self.plot_returns_distribution(ax2)
        
        # 3. Portfolio Evolution
        ax3 = plt.subplot(2, 3, 3)
        self.plot_portfolio_evolution(ax3)
        
        # 4. Trading Frequency Analysis
        ax4 = plt.subplot(2, 3, 4)
        self.plot_trading_frequency(ax4)
        
        # 5. ETF Allocation
        ax5 = plt.subplot(2, 3, 5)
        self.plot_etf_allocation(ax5)
        
        # 6. Risk-Return Analysis
        ax6 = plt.subplot(2, 3, 6)
        self.plot_risk_return_analysis(ax6)
        
        plt.tight_layout()
        plt.savefig('ml_strategy_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Charts saved to: ml_strategy_analysis.png")
        
        plt.show()
    
    def plot_model_comparison(self, ax):
        """Plot model performance comparison"""
        models = []
        outperformance = []
        
        for model_name, results in self.results.items():
            if 'outperformance' in results:
                models.append(model_name.upper())
                outperformance.append(results['outperformance'])
        
        if models:
            colors = ['green' if x > 0 else 'red' for x in outperformance]
            bars = ax.bar(models, outperformance, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, outperformance):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
            
            ax.set_title('Model Outperformance vs SPY', fontsize=14, fontweight='bold')
            ax.set_ylabel('Outperformance (%)')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    def plot_returns_distribution(self, ax):
        """Plot returns distribution"""
        all_returns = []
        
        for model_name, results in self.results.items():
            if 'strategy_return' in results:
                all_returns.append(results['strategy_return'])
        
        if all_returns:
            ax.hist(all_returns, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=np.mean(all_returns), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_returns):.1f}%')
            ax.set_title('Strategy Returns Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Strategy Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
    
    def plot_portfolio_evolution(self, ax):
        """Plot portfolio value evolution"""
        best_model = None
        best_performance = -999999
        
        # Find best performing model
        for model_name, results in self.results.items():
            if 'outperformance' in results and results['outperformance'] > best_performance:
                best_performance = results['outperformance']
                best_model = results
        
        if best_model and 'portfolio_history' in best_model:
            portfolio_values = best_model['portfolio_history']
            spy_values = best_model.get('spy_history', [])
            
            x = range(len(portfolio_values))
            ax.plot(x, portfolio_values, label='Strategy', linewidth=2, color='green')
            
            if spy_values:
                ax.plot(x, spy_values, label='SPY Benchmark', linewidth=2, color='blue', alpha=0.7)
            
            ax.set_title('Portfolio Value Evolution (Best Model)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def plot_trading_frequency(self, ax):
        """Plot trading frequency analysis"""
        trade_counts = []
        model_names = []
        
        for model_name, results in self.results.items():
            if 'total_trades' in results:
                model_names.append(model_name.upper())
                trade_counts.append(results['total_trades'])
        
        if model_names:
            bars = ax.bar(model_names, trade_counts, color='orange', alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, trade_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value}', ha='center', va='bottom')
            
            ax.set_title('Trading Frequency by Model', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Trades')
            ax.tick_params(axis='x', rotation=45)
    
    def plot_etf_allocation(self, ax):
        """Plot ETF allocation analysis"""
        # Get ETF allocation from best model
        best_model = None
        best_performance = -999999
        
        for model_name, results in self.results.items():
            if 'outperformance' in results and results['outperformance'] > best_performance:
                best_performance = results['outperformance']
                best_model = results
        
        if best_model and 'trades' in best_model:
            etf_counts = {}
            for trade in best_model['trades']:
                etf = trade.get('to_etf', 'Unknown')
                etf_counts[etf] = etf_counts.get(etf, 0) + 1
            
            if etf_counts:
                etfs = list(etf_counts.keys())
                counts = list(etf_counts.values())
                
                wedges, texts, autotexts = ax.pie(counts, labels=etfs, autopct='%1.1f%%', startangle=90)
                ax.set_title('ETF Allocation (Best Model)', fontsize=14, fontweight='bold')
    
    def plot_risk_return_analysis(self, ax):
        """Plot risk-return scatter analysis"""
        returns = []
        risks = []  # Using number of trades as risk proxy
        model_names = []
        
        for model_name, results in self.results.items():
            if 'strategy_return' in results and 'total_trades' in results:
                returns.append(results['strategy_return'])
                risks.append(results['total_trades'])
                model_names.append(model_name.upper())
        
        if returns:
            scatter = ax.scatter(risks, returns, s=100, alpha=0.7, c=range(len(returns)), cmap='viridis')
            
            # Add model labels
            for i, name in enumerate(model_names):
                ax.annotate(name, (risks[i], returns[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
            
            ax.set_title('Risk-Return Analysis', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Trades (Risk Proxy)')
            ax.set_ylabel('Strategy Return (%)')
            ax.grid(True, alpha=0.3)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive HTML report"""
        print(f"\n📝 GENERATING COMPREHENSIVE REPORT...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML ETF Strategy Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #2980b9; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .success {{ background-color: #d5f4e6; border-left: 5px solid #27ae60; }}
                .warning {{ background-color: #fdf2e9; border-left: 5px solid #e67e22; }}
                .error {{ background-color: #fadbd8; border-left: 5px solid #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .chart-container {{ text-align: center; margin: 30px 0; }}
                .timestamp {{ color: #7f8c8d; font-size: 12px; text-align: center; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 ML ETF Strategy Analysis Report</h1>
                
                <div class="metric success">
                    <h3>📊 Executive Summary</h3>
                    <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Models Analyzed:</strong> {len(self.results)}</p>
                    <p><strong>Status:</strong> Analysis Complete ✅</p>
                </div>
        """
        
        # Add model performance table
        html_content += """
                <h2>🏆 Model Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Strategy Return</th>
                        <th>SPY Return</th>
                        <th>Outperformance</th>
                        <th>Total Trades</th>
                        <th>Status</th>
                    </tr>
        """
        
        for model_name, results in self.results.items():
            if 'strategy_return' in results:
                status_class = 'success' if results.get('outperformance', 0) > 0 else 'error'
                status_text = '🏆 SUCCESS' if results.get('outperformance', 0) > 0 else '⚠️ UNDERPERFORMED'
                
                html_content += f"""
                    <tr class="{status_class}">
                        <td><strong>{model_name.upper()}</strong></td>
                        <td>{results.get('strategy_return', 0):+.2f}%</td>
                        <td>{results.get('spy_return', 0):+.2f}%</td>
                        <td>{results.get('outperformance', 0):+.2f}%</td>
                        <td>{results.get('total_trades', 0)}</td>
                        <td>{status_text}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <div class="chart-container">
                    <h3>📈 Performance Visualization</h3>
                    <img src="ml_strategy_analysis.png" alt="ML Strategy Analysis Charts" style="max-width: 100%; height: auto;">
                </div>
                
                <div class="timestamp">
                    Report generated by ML Result Viewer | ETF-app-1w Project
                </div>
            </div>
        </body>
        </html>
        """
        
        with open('ml_analysis_report.html', 'w') as f:
            f.write(html_content)
        
        print("✅ Comprehensive report saved to: ml_analysis_report.html")
    
    def run_complete_analysis(self):
        """Run complete ML result analysis"""
        print("🚀 STARTING COMPREHENSIVE ML ANALYSIS")
        print("=" * 60)
        
        # Display all phases
        self.display_training_summary()
        self.display_validation_analysis()
        self.display_detailed_testing_results()
        
        # Create visualizations
        self.create_performance_charts()
        
        # Generate report
        self.generate_comprehensive_report()
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print("Files generated:")
        print("  📊 ml_strategy_analysis.png - Performance charts")
        print("  📄 ml_analysis_report.html - Comprehensive report")
        print("  📋 Console output - Detailed analysis")

def main():
    """Main execution function"""
    viewer = MLResultViewer()
    viewer.run_complete_analysis()

if __name__ == "__main__":
    main()