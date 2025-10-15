"""
Add sample QQQ and IWM data to the existing database for 3-ETF rotation demo
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def add_sample_etf_data():
    """Add sample QQQ and IWM data based on SPY patterns"""
    conn = sqlite3.connect('etf_data.db')
    
    # Get SPY data as reference
    spy_df = pd.read_sql_query("SELECT * FROM prices WHERE symbol = 'SPY' ORDER BY date", conn)
    
    if spy_df.empty:
        print("❌ No SPY data found to base sample data on")
        return
    
    print(f"📊 Found {len(spy_df)} SPY records")
    
    # Create QQQ data (tech-heavy, more volatile)
    print("🔧 Creating QQQ sample data...")
    qqq_data = []
    
    for _, row in spy_df.iterrows():
        # QQQ typically correlates with SPY but with higher volatility
        # Add some random variation to make it realistic
        base_multiplier = 1.2 + random.uniform(-0.3, 0.3)  # QQQ often trades higher than SPY
        volatility_factor = 1.5  # Higher volatility
        
        qqq_close = row['close'] * base_multiplier * (1 + random.uniform(-0.02, 0.02) * volatility_factor)
        qqq_sma_5d = qqq_close * (1 + random.uniform(-0.01, 0.01))
        
        qqq_data.append({
            'symbol': 'QQQ',
            'date': row['date'],
            'close': round(qqq_close, 2),
            'sma_5d': round(qqq_sma_5d, 2),
            'monthly_trend': random.choice([1, -1])  # Random trend
        })
    
    # Create IWM data (small cap, different patterns)
    print("🔧 Creating IWM sample data...")
    iwm_data = []
    
    for _, row in spy_df.iterrows():
        # IWM typically lower price, different volatility pattern
        base_multiplier = 0.6 + random.uniform(-0.2, 0.2)  # IWM often trades lower than SPY
        volatility_factor = 1.3  # Different volatility pattern
        
        iwm_close = row['close'] * base_multiplier * (1 + random.uniform(-0.02, 0.02) * volatility_factor)
        iwm_sma_5d = iwm_close * (1 + random.uniform(-0.01, 0.01))
        
        iwm_data.append({
            'symbol': 'IWM',
            'date': row['date'],
            'close': round(iwm_close, 2),
            'sma_5d': round(iwm_sma_5d, 2),
            'monthly_trend': random.choice([1, -1])  # Random trend
        })
    
    # Insert QQQ data
    cursor = conn.cursor()
    
    # Check if QQQ data already exists
    cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = 'QQQ'")
    qqq_count = cursor.fetchone()[0]
    
    if qqq_count == 0:
        print("📥 Inserting QQQ data...")
        for data in qqq_data:
            cursor.execute("""
                INSERT INTO prices (symbol, date, close, sma_5d, monthly_trend)
                VALUES (?, ?, ?, ?, ?)
            """, (data['symbol'], data['date'], data['close'], data['sma_5d'], data['monthly_trend']))
        print(f"✅ Inserted {len(qqq_data)} QQQ records")
    else:
        print(f"⚠️ QQQ data already exists ({qqq_count} records)")
    
    # Check if IWM data already exists
    cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = 'IWM'")
    iwm_count = cursor.fetchone()[0]
    
    if iwm_count == 0:
        print("📥 Inserting IWM data...")
        for data in iwm_data:
            cursor.execute("""
                INSERT INTO prices (symbol, date, close, sma_5d, monthly_trend)
                VALUES (?, ?, ?, ?, ?)
            """, (data['symbol'], data['date'], data['close'], data['sma_5d'], data['monthly_trend']))
        print(f"✅ Inserted {len(iwm_data)} IWM records")
    else:
        print(f"⚠️ IWM data already exists ({iwm_count} records)")
    
    conn.commit()
    
    # Verify final data
    cursor.execute("SELECT symbol, COUNT(*) as count FROM prices GROUP BY symbol ORDER BY symbol")
    results = cursor.fetchall()
    
    print("\n📊 Final ETF data summary:")
    for symbol, count in results:
        print(f"  {symbol}: {count} records")
    
    conn.close()
    print("✅ ETF data preparation complete!")

if __name__ == "__main__":
    add_sample_etf_data()