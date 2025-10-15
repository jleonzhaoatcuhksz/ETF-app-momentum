"""
Remove QQQ and IWM sample data from the database
"""

import sqlite3

def remove_sample_etfs():
    """Remove QQQ and IWM data from the database"""
    conn = sqlite3.connect('etf_data.db')
    cursor = conn.cursor()
    
    print("🗑️ Removing QQQ and IWM sample data...")
    
    # Check current data
    cursor.execute("SELECT symbol, COUNT(*) as count FROM prices GROUP BY symbol ORDER BY symbol")
    before_results = cursor.fetchall()
    
    print("📊 Before removal:")
    for symbol, count in before_results:
        print(f"  {symbol}: {count} records")
    
    # Remove QQQ data
    cursor.execute("DELETE FROM prices WHERE symbol = 'QQQ'")
    qqq_deleted = cursor.rowcount
    print(f"✅ Deleted {qqq_deleted} QQQ records")
    
    # Remove IWM data
    cursor.execute("DELETE FROM prices WHERE symbol = 'IWM'")
    iwm_deleted = cursor.rowcount
    print(f"✅ Deleted {iwm_deleted} IWM records")
    
    conn.commit()
    
    # Check final data
    cursor.execute("SELECT symbol, COUNT(*) as count FROM prices GROUP BY symbol ORDER BY symbol")
    after_results = cursor.fetchall()
    
    print("\n📊 After removal:")
    for symbol, count in after_results:
        print(f"  {symbol}: {count} records")
    
    conn.close()
    print("✅ Sample ETF data removal complete!")

if __name__ == "__main__":
    remove_sample_etfs()