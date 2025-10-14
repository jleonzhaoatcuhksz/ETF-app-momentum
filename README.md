# ETF-app-data

A comprehensive ETF data collection system that retrieves historical closing prices for 14 core ETFs from 2015-2025 and stores them in an SQLite database.

## 📊 Covered ETFs

### Core ETFs (3)
- **SPY** - SPDR S&P 500 ETF Trust
- **TLT** - iShares 20+ Year Treasury Bond ETF  
- **SHY** - iShares 1-3 Year Treasury Bond ETF

### Sector ETFs (11)
- **XLK** - Technology Select Sector SPDR Fund
- **XLF** - Financial Select Sector SPDR Fund
- **XLV** - Health Care Select Sector SPDR Fund
- **XLI** - Industrial Select Sector SPDR Fund
- **XLE** - Energy Select Sector SPDR Fund
- **XLY** - Consumer Discretionary Select Sector SPDR Fund
- **XLP** - Consumer Staples Select Sector SPDR Fund
- **XLU** - Utilities Select Sector SPDR Fund
- **XLB** - Materials Select Sector SPDR Fund
- **XLRE** - Real Estate Select Sector SPDR Fund
- **XLC** - Communication Services Select Sector SPDR Fund

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Demo (Recommended First)
```bash
npm run demo
```
- Tests the system with recent data (last 30 days)
- Verifies Yahoo Finance API connectivity
- Creates sample database

### 3. Fetch Full Historical Data
```bash
npm run fetch
```
- Downloads 10+ years of historical data (2015-2025)
- Processes ~36,500 price records per ETF
- Takes 15-30 minutes due to API rate limiting

### 4. Start Web Interface
```bash
npm run server
```
- Launches web interface at http://localhost:3019
- Browse ETFs and historical price data
- Interactive charts and data export

## 📁 Project Structure

```
ETF-app-data/
├── config.js          # Configuration (ETFs, date ranges, API settings)
├── database.js        # SQLite database operations
├── yahooFinance.js    # Yahoo Finance API client
├── fetchData.js       # Main data collection script
├── logger.js          # Logging utilities
├── demo.js           # Demo/test script
├── server.js         # Web interface server
├── package.json      # Dependencies and scripts
└── README.md         # This file
```

## 🗄️ Database Schema

### ETFs Table
```sql
CREATE TABLE etfs (
    symbol TEXT PRIMARY KEY,     -- ETF ticker (e.g., 'SPY')
    name TEXT NOT NULL,          -- Full ETF name
    sector TEXT NOT NULL,        -- Sector/category
    category TEXT NOT NULL,      -- 'Core ETF' or 'Sector ETF'
    created_at DATETIME,         -- Record creation time
    updated_at DATETIME          -- Last update time
);
```

### Prices Table
```sql
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,        -- ETF ticker (foreign key)
    date TEXT NOT NULL,          -- Price date (YYYY-MM-DD)
    open REAL,                   -- Opening price
    high REAL,                   -- Daily high
    low REAL,                    -- Daily low
    close REAL NOT NULL,         -- Closing price
    volume INTEGER,              -- Trading volume
    adj_close REAL,              -- Adjusted closing price
    created_at DATETIME,         -- Record creation time
    UNIQUE(symbol, date)         -- Prevents duplicates
);
```

## ⚙️ Configuration

Edit `config.js` to customize:

- **Date Range**: Modify `dateRange.startDate` and `endDate`
- **API Settings**: Adjust request delays and batch sizes
- **ETF List**: Add/remove ETFs from the `etfs` array
- **Database Path**: Change database file location

## 📈 Expected Data Volume

For 10+ years (2015-2025) of daily data:
- **~2,600 trading days** per ETF
- **14 ETFs** × 2,600 days = **~36,400 total records**
- **Database size**: ~15-20 MB
- **Fetch time**: 15-30 minutes (with rate limiting)

## 🔧 API Rate Limiting

The system includes built-in rate limiting to avoid Yahoo Finance API restrictions:
- 300ms delay between individual requests
- Processes 3 ETFs simultaneously in batches
- Automatic retry with exponential backoff
- Comprehensive error handling

## 📊 Features

- **Robust Data Collection**: Handles API failures gracefully
- **Duplicate Prevention**: Unique constraints prevent data duplication
- **Progress Tracking**: Real-time progress indicators
- **Comprehensive Logging**: Detailed logs with timestamps
- **Performance Optimized**: Indexes for fast queries
- **Web Interface**: Browse and visualize data
- **Export Capabilities**: CSV export functionality

## 🚨 Important Notes

1. **Yahoo Finance API**: Free but rate-limited. Respect the delays.
2. **Data Gaps**: Weekends and holidays have no trading data.
3. **Market Hours**: Only includes regular trading session data.
4. **Adjusted Prices**: Includes dividend and split adjustments.
5. **Storage**: Ensure sufficient disk space (~20MB minimum).

## 🛠️ Troubleshooting

### Common Issues:

**"Failed to fetch data"**
- Check internet connectivity
- Yahoo Finance may be temporarily unavailable
- Try running demo first to test connectivity

**"Database locked"**
- Only run one instance at a time
- Database connection will auto-close after completion

**"API rate limit exceeded"**
- Increase delays in config.js
- Reduce batch size
- Wait and retry later

## 📝 Usage Examples

```javascript
// Get price data for SPY
const prices = await database.getPriceData('SPY', '2020-01-01', '2020-12-31');

// Get all ETFs
const etfs = await database.getETFList();

// Get database statistics
const stats = await database.getStats();
```

## 🎯 Next Steps

After successful data collection:
1. Explore data using the web interface
2. Build trading strategies with historical data
3. Implement backtesting algorithms
4. Create performance analytics
5. Set up automated daily updates

---

**Happy ETF data analysis! 📈**