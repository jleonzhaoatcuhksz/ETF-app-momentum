"""
3-ETF Rotation Strategy using XGBoost/LightGBM
Rotating between SPY, QQQ, and IWM based on ML predictions
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class ThreeETFRotationStrategy:
    def __init__(self, db_path='./etf_data.db'):
        """3-ETF rotation using gradient boosting (SPY, QQQ, IWM)"""
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
        # Target ETFs for rotation
        self.target_etfs = ['SPY', 'QQQ', 'IWM']
        
        print("🔄 3-ETF ROTATION STRATEGY")
        print("Target ETFs: SPY (Large Cap), QQQ (Tech), IWM (Small Cap)")
        print("=" * 60)
        
    def load_data(self):
        """Load data for all 3 target ETFs"""
        conn = sqlite3.connect(self.db_path)
        
        # Load data for all target ETFs
        query = """
        SELECT symbol, date, close, sma_5d, monthly_trend,
               LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
               LAG(close, 5) OVER (PARTITION BY symbol ORDER BY date) as close_5d_ago,
               LAG(close, 20) OVER (PARTITION BY symbol ORDER BY date) as close_20d_ago
        FROM prices 
        WHERE symbol IN ('SPY', 'QQQ', 'IWM')
        ORDER BY symbol, date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No data found for target ETFs")
            return None
            
        print(f"✅ Loaded {len(df)} records for {df['symbol'].nunique()} ETFs")
        print(f"ETFs found: {sorted(df['symbol'].unique())}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    def create_features(self, df):
        """Create features for 3-ETF rotation prediction"""
        print("🔧 Creating features for 3-ETF rotation...")
        
        # Calculate returns and technical indicators
        df['return_1d'] = (df['close'] - df['prev_close']) / df['prev_close']
        df['return_5d'] = (df['close'] - df['close_5d_ago']) / df['close_5d_ago']
        df['return_20d'] = (df['close'] - df['close_20d_ago']) / df['close_20d_ago']
        
        # SMA ratio
        df['sma_ratio'] = df['close'] / df['sma_5d']
        
        # Volatility (rolling standard deviation)
        df = df.sort_values(['symbol', 'date'])
        df['volatility'] = df.groupby('symbol')['return_1d'].rolling(20).std().reset_index(0, drop=True)
        
        # Create relative performance features
        pivot_df = df.pivot(index='date', columns='symbol', values='return_1d')
        
        # Calculate relative performance vs other ETFs
        for etf in self.target_etfs:
            if etf in pivot_df.columns:
                other_etfs = [e for e in self.target_etfs if e != etf and e in pivot_df.columns]
                if other_etfs:
                    pivot_df[f'{etf}_vs_others'] = pivot_df[etf] - pivot_df[other_etfs].mean(axis=1)
        
        # Merge back relative performance
        rel_perf_df = pivot_df[[col for col in pivot_df.columns if '_vs_others' in col]].reset_index()
        df = df.merge(rel_perf_df, on='date', how='left')
        
        # Create target: which ETF will perform best in next period
        future_returns = df.groupby('symbol')['return_1d'].shift(-1)
        df['future_return'] = future_returns
        
        # For each date, determine which ETF has the highest future return
        daily_best = df.groupby('date')['future_return'].transform('max')
        df['is_best_performer'] = (df['future_return'] == daily_best).astype(int)
        
        # Remove rows with missing values
        feature_cols = ['return_1d', 'return_5d', 'return_20d', 'sma_ratio', 'volatility', 'monthly_trend']
        feature_cols += [col for col in df.columns if '_vs_others' in col]
        
        df_clean = df.dropna(subset=feature_cols + ['future_return'])
        
        print(f"✅ Created {len(feature_cols)} features")
        print(f"✅ Clean dataset: {len(df_clean)} records")
        
        return df_clean, feature_cols
        
    def train_models(self, df, feature_cols):
        """Train models for 3-ETF rotation prediction"""
        print("🚀 Training 3-ETF rotation models...")
        
        # Prepare features and targets
        X = df[feature_cols].values
        y = df['is_best_performer'].values
        symbols = df['symbol'].values
        dates = pd.to_datetime(df['date'])
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        results = {}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            fold_results = {}
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            rf_prob = rf.predict_proba(X_test_scaled)[:, 1]
            
            fold_results['random_forest'] = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'predictions': rf_pred.tolist(),
                'probabilities': rf_prob.tolist()
            }
            
            # XGBoost
            xgb_model = xgb.XGBClassifier(random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
            
            fold_results['xgboost'] = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'predictions': xgb_pred.tolist(),
                'probabilities': xgb_prob.tolist()
            }
            
            # LightGBM
            lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
            lgb_model.fit(X_train_scaled, y_train)
            lgb_pred = lgb_model.predict(X_test_scaled)
            lgb_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]
            
            fold_results['lightgbm'] = {
                'accuracy': accuracy_score(y_test, lgb_pred),
                'predictions': lgb_pred.tolist(),
                'probabilities': lgb_prob.tolist()
            }
            
            # Store test data info
            test_dates = dates.iloc[test_idx]
            test_symbols = symbols[test_idx]
            
            fold_results['test_info'] = {
                'dates': [d.strftime('%Y-%m-%d') for d in test_dates],
                'symbols': test_symbols.tolist(),
                'actual_labels': y_test.tolist()
            }
            
            results[f'fold_{fold}'] = fold_results
            
        # Calculate overall performance
        overall_results = {}
        for model in ['random_forest', 'xgboost', 'lightgbm']:
            accuracies = [results[f'fold_{i}'][model]['accuracy'] for i in range(3)]
            overall_results[model] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'fold_accuracies': accuracies
            }
            
        results['overall_performance'] = overall_results
        
        print("✅ Model training completed!")
        for model, perf in overall_results.items():
            print(f"{model}: {perf['mean_accuracy']:.3f} ± {perf['std_accuracy']:.3f}")
            
        return results
        
    def simulate_rotation_strategy(self, df, results):
        """Simulate 3-ETF rotation trading strategy"""
        print("📈 Simulating 3-ETF rotation strategy...")
        
        # Get the best model based on mean accuracy
        best_model = max(results['overall_performance'].items(), 
                        key=lambda x: x[1]['mean_accuracy'])[0]
        
        print(f"Using best model: {best_model}")
        
        # Simulate trading based on predictions
        portfolio_value = 10000  # Starting value
        portfolio_history = []
        current_position = None
        
        # Combine all test predictions
        all_predictions = []
        all_dates = []
        all_symbols = []
        
        for fold in range(3):
            fold_data = results[f'fold_{fold}']
            predictions = fold_data[best_model]['probabilities']
            dates = fold_data['test_info']['dates']
            symbols = fold_data['test_info']['symbols']
            
            for i, (date, symbol, prob) in enumerate(zip(dates, symbols, predictions)):
                all_predictions.append({
                    'date': date,
                    'symbol': symbol,
                    'probability': prob
                })
        
        # Group by date and find best ETF for each date
        pred_df = pd.DataFrame(all_predictions)
        daily_best = pred_df.groupby('date').apply(
            lambda x: x.loc[x['probability'].idxmax(), 'symbol']
        ).reset_index()
        daily_best.columns = ['date', 'best_etf']
        
        # Calculate strategy returns
        strategy_returns = []
        spy_returns = []  # Benchmark
        
        for _, row in daily_best.iterrows():
            date = row['date']
            best_etf = row['best_etf']
            
            # Get actual returns for this date
            date_data = df[df['date'] == date]
            if not date_data.empty:
                # Get strategy return (best ETF)
                etf_data = date_data[date_data['symbol'] == best_etf]['return_1d']
                if not etf_data.empty:
                    etf_return = etf_data.iloc[0]
                    strategy_returns.append(etf_return)
                    
                    # Get SPY benchmark return
                    spy_data = date_data[date_data['symbol'] == 'SPY']['return_1d']
                    if not spy_data.empty:
                        spy_return = spy_data.iloc[0]
                        spy_returns.append(spy_return)
                    else:
                        # If no SPY data, use the strategy return as benchmark
                        spy_returns.append(etf_return)
        
        # Calculate cumulative returns
        strategy_cumret = np.cumprod(1 + np.array(strategy_returns)) - 1
        spy_cumret = np.cumprod(1 + np.array(spy_returns)) - 1
        
        trading_results = {
            'strategy_return': strategy_cumret[-1] if len(strategy_cumret) > 0 else 0,
            'spy_return': spy_cumret[-1] if len(spy_cumret) > 0 else 0,
            'outperformance': strategy_cumret[-1] - spy_cumret[-1] if len(strategy_cumret) > 0 else 0,
            'num_trades': len(strategy_returns),
            'best_model_used': best_model,
            'daily_selections': daily_best.to_dict('records')
        }
        
        print(f"✅ Strategy Return: {trading_results['strategy_return']:.2%}")
        print(f"✅ SPY Return: {trading_results['spy_return']:.2%}")
        print(f"✅ Outperformance: {trading_results['outperformance']:.2%}")
        
        return trading_results
        
    def run_analysis(self):
        """Run complete 3-ETF rotation analysis"""
        # Load data
        df = self.load_data()
        if df is None:
            return None
            
        # Create features
        df_features, feature_cols = self.create_features(df)
        
        # Train models
        model_results = self.train_models(df_features, feature_cols)
        
        # Simulate trading
        trading_results = self.simulate_rotation_strategy(df_features, model_results)
        
        # Combine results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'strategy_type': '3_etf_rotation',
            'target_etfs': self.target_etfs,
            'model_performance': model_results['overall_performance'],
            'trading_performance': trading_results,
            'feature_count': len(feature_cols),
            'data_points': len(df_features)
        }
        
        # Save results
        with open('3etf_rotation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
            
        print(f"✅ Results saved to 3etf_rotation_results.json")
        return final_results

if __name__ == "__main__":
    strategy = ThreeETFRotationStrategy()
    results = strategy.run_analysis()
    
    if results:
        print("\n" + "="*60)
        print("3-ETF ROTATION STRATEGY RESULTS")
        print("="*60)
        print(f"Strategy Return: {results['trading_performance']['strategy_return']:.2%}")
        print(f"SPY Benchmark: {results['trading_performance']['spy_return']:.2%}")
        print(f"Outperformance: {results['trading_performance']['outperformance']:.2%}")
        print(f"Number of Trades: {results['trading_performance']['num_trades']}")
        print(f"Best Model: {results['trading_performance']['best_model_used']}")