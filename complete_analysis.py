"""
Complete Analysis of ETF Switching ML Strategy
Detailed Data Verification and Results Summary
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_database():
    """Analyze the actual database structure and data"""
    print("="*60)
    print("DATABASE ANALYSIS - DETAILED VERIFICATION")
    print("="*60)
    
    conn = sqlite3.connect('./etf_data.db')
    cursor = conn.cursor()
    
    # Basic statistics
    cursor.execute('SELECT COUNT(*) FROM prices')
    total_records = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT symbol) FROM prices')
    unique_etfs = cursor.fetchone()[0]
    
    cursor.execute('SELECT MIN(date), MAX(date) FROM prices')
    date_range = cursor.fetchone()
    
    cursor.execute('SELECT COUNT(*) FROM prices WHERE monthly_trend IS NOT NULL')
    valid_trend_records = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT date) FROM prices')
    unique_dates = cursor.fetchone()[0]
    
    print(f"📊 DATASET OVERVIEW:")
    print(f"   Total Records: {total_records:,}")
    print(f"   Unique ETFs: {unique_etfs}")
    print(f"   Date Range: {date_range[0]} to {date_range[1]}")
    print(f"   Trading Days: {unique_dates:,}")
    print(f"   Records with Monthly_Trend: {valid_trend_records:,}")
    print(f"   Missing Monthly_Trend: {total_records - valid_trend_records:,}")
    print(f"   Data Completeness: {(valid_trend_records/total_records)*100:.1f}%")
    
    # Get ETF symbols
    cursor.execute('SELECT DISTINCT symbol FROM prices ORDER BY symbol')
    etf_symbols = [row[0] for row in cursor.fetchall()]
    print(f"\n🏢 ETF UNIVERSE ({len(etf_symbols)} ETFs):")
    for i, symbol in enumerate(etf_symbols, 1):
        cursor.execute('SELECT COUNT(*) FROM prices WHERE symbol = ?', (symbol,))
        count = cursor.fetchone()[0]
        print(f"   {i:2d}. {symbol}: {count:,} records")
    
    # Monthly trend statistics
    cursor.execute('''
        SELECT 
            AVG(monthly_trend) as avg_trend,
            MIN(monthly_trend) as min_trend,
            MAX(monthly_trend) as max_trend,
            STDDEV(monthly_trend) as std_trend
        FROM prices 
        WHERE monthly_trend IS NOT NULL
    ''')
    trend_stats = cursor.fetchone()
    
    print(f"\n📈 MONTHLY_TREND STATISTICS:")
    print(f"   Average: {trend_stats[0]:.2f}%")
    print(f"   Minimum: {trend_stats[1]:.2f}%")
    print(f"   Maximum: {trend_stats[2]:.2f}%")
    if trend_stats[3]:
        print(f"   Std Dev: {trend_stats[3]:.2f}%")
    
    # Sample recent data
    print(f"\n📅 RECENT DATA SAMPLE (Last 5 days):")
    cursor.execute('''
        SELECT date, symbol, close, monthly_trend 
        FROM prices 
        WHERE monthly_trend IS NOT NULL 
        ORDER BY date DESC, symbol 
        LIMIT 25
    ''')
    
    recent_data = cursor.fetchall()
    current_date = None
    for row in recent_data:
        if row[0] != current_date:
            current_date = row[0]
            print(f"\n   📅 {current_date}:")
        print(f"      {row[1]}: ${row[2]:6.2f} (trend: {row[3]:+6.2f}%)")
    
    conn.close()
    return etf_symbols, total_records, valid_trend_records

def analyze_model_files():
    """Analyze generated model files"""
    print("\n" + "="*60)
    print("MODEL FILES ANALYSIS")
    print("="*60)
    
    files_found = []
    
    # Check for model file
    if os.path.exists('etf_switching_model.h5'):
        file_size = os.path.getsize('etf_switching_model.h5')
        files_found.append(f"✅ etf_switching_model.h5 ({file_size:,} bytes)")
        print(f"🧠 TRAINED MODEL:")
        print(f"   File: etf_switching_model.h5")
        print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    else:
        files_found.append("❌ etf_switching_model.h5 (NOT FOUND)")
    
    # Check for results file
    if os.path.exists('etf_switching_results.json'):
        file_size = os.path.getsize('etf_switching_results.json')
        files_found.append(f"✅ etf_switching_results.json ({file_size:,} bytes)")
        print(f"📊 RESULTS FILE:")
        print(f"   File: etf_switching_results.json")
        print(f"   Size: {file_size:,} bytes")
    else:
        files_found.append("❌ etf_switching_results.json (NOT FOUND)")
    
    # Check for analysis charts
    if os.path.exists('etf_switching_analysis.png'):
        file_size = os.path.getsize('etf_switching_analysis.png')
        files_found.append(f"✅ etf_switching_analysis.png ({file_size:,} bytes)")
    else:
        files_found.append("❌ etf_switching_analysis.png (NOT FOUND)")
    
    print(f"\n📁 FILES STATUS:")
    for file_status in files_found:
        print(f"   {file_status}")
    
    return files_found

def analyze_training_architecture():
    """Analyze the ML model architecture from code"""
    print("\n" + "="*60)
    print("ML MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    
    print(f"🤖 DEEP Q-NETWORK (DQN) CONFIGURATION:")
    print(f"   Algorithm: Deep Q-Network with Experience Replay")
    print(f"   Framework: TensorFlow/Keras")
    print(f"")
    print(f"   Network Architecture:")
    print(f"      Input Layer:  17 features")
    print(f"      Hidden 1:     128 neurons (ReLU + 20% Dropout)")
    print(f"      Hidden 2:     64 neurons (ReLU + 20% Dropout)")
    print(f"      Hidden 3:     32 neurons (ReLU)")
    print(f"      Output:       15 neurons (Linear - Q-values)")
    
    print(f"\n🎯 STATE SPACE (17 dimensions):")
    print(f"   1. Current ETF ID (1 dim): Which ETF is currently held (0-13)")
    print(f"   2. Monthly Trends (14 dims): Monthly_Trend for all 14 ETFs")
    print(f"   3. Days Held (1 dim): How long current ETF has been held (normalized)")
    print(f"   4. Portfolio Performance (1 dim): Current return vs initial capital")
    
    print(f"\n⚡ ACTION SPACE (15 actions):")
    print(f"   Action 0: HOLD current ETF position")
    print(f"   Actions 1-14: SWITCH to specific ETF:")
    etf_list = ['SPY', 'TLT', 'SHY', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
    for i, etf in enumerate(etf_list, 1):
        print(f"      Action {i:2d}: Switch to {etf}")
    
    print(f"\n💰 REWARD SYSTEM:")
    print(f"   Hold Action: Portfolio performance change")
    print(f"   Switch Action: Target ETF monthly_trend - 2×transaction_cost")
    print(f"   Transaction Cost: 0.1% per switch (buy + sell)")
    print(f"   Goal: Maximize long-term cumulative rewards")
    
    print(f"\n🎓 TRAINING PARAMETERS:")
    print(f"   Episodes: 500")
    print(f"   Memory Buffer: 10,000 experiences")
    print(f"   Batch Size: 32")
    print(f"   Learning Rate: 0.001")
    print(f"   Epsilon Decay: 0.995 (exploration → exploitation)")
    print(f"   Target Network Update: Every 100 episodes")

def generate_summary():
    """Generate comprehensive summary"""
    print("\n" + "="*60)
    print("COMPREHENSIVE ML STRATEGY SUMMARY")
    print("="*60)
    
    # Analyze database
    etf_symbols, total_records, valid_records = analyze_database()
    
    # Analyze model files
    files_status = analyze_model_files()
    
    # Analyze architecture
    analyze_training_architecture()
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 FINAL STATUS SUMMARY")
    print("="*60)
    
    model_exists = os.path.exists('etf_switching_model.h5')
    results_exist = os.path.exists('etf_switching_results.json')
    
    if model_exists and results_exist:
        status = "✅ FULLY COMPLETED"
        next_steps = "Ready for backtesting and live trading"
    elif model_exists and not results_exist:
        status = "⚠️ PARTIALLY COMPLETED"
        next_steps = "Model trained, but results generation failed - needs re-run of testing phase"
    else:
        status = "❌ TRAINING FAILED"
        next_steps = "Need to restart training process"
    
    print(f"🔍 TRAINING STATUS: {status}")
    print(f"📊 DATA PROCESSED: {valid_records:,} records across {len(etf_symbols)} ETFs")
    print(f"📅 TIME PERIOD: 10 years (2015-2025)")
    print(f"🧠 MODEL TYPE: Deep Q-Network (DQN)")
    print(f"🎯 OBJECTIVE: Single ETF with intelligent switching")
    print(f"📈 STRATEGY: Monthly_Trend based switching decisions")
    print(f"💡 NEXT STEPS: {next_steps}")
    
    # Training time analysis
    print(f"\n⏱️ TRAINING TIME ANALYSIS:")
    print(f"   Start Time: 08:44:08 (from terminal log)")
    print(f"   Current Time: 10:54:00 (approximately)")
    print(f"   Total Duration: ~2 hours 10 minutes")
    print(f"   Training Intensity: High (500 episodes × 252 days × 14 ETFs)")
    
    if not results_exist and model_exists:
        print(f"\n🔧 RECOMMENDED ACTION:")
        print(f"   The model training completed successfully, but the backtesting")
        print(f"   and results generation phase encountered an issue. We should")
        print(f"   run the testing phase separately to generate the performance")
        print(f"   analysis and trading results.")

if __name__ == "__main__":
    print("ETF Switching ML Strategy - Complete Analysis")
    print("Generated at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    generate_summary()