# ETF-App-3w - 3-ETF Rotation Strategy

## 🎯 MULTI-ETF ROTATION ARCHITECTURE

```
ETF-app-3w/
├── 🌐 WEB APPLICATION
│   ├── server_clean.js          # Main Express server (PORT: 3023)
│   ├── package.json             # Node.js dependencies  
│   ├── package-lock.json        # Dependency lock file
│   └── public/                  # Static web files
│
├── 📊 DATABASE & DATA
│   ├── etf_data.db             # Main SQLite database (3 ETFs: SPY, QQQ, IWM)
│   ├── etf_data.db-shm         # SQLite shared memory
│   └── etf_data.db-wal         # SQLite write-ahead log
│
├── 🤖 ML STRATEGY ENGINE (3-ETF ROTATION)
│   ├── fast_ml_strategy_3etf.py # 🔥 MAIN 3-ETF rotation strategy
│   └── ml_result_viewer_fixed.py # Results analyzer & viewer
│
├── 📈 ML RESULTS (JSON) - 3-ETF ROTATION
│   ├── random_forest_3etf_results.json    # Random Forest 3-ETF results
│   ├── lightgbm_3etf_results.json        # LightGBM 3-ETF results  
│   ├── xgboost_3etf_results.json         # XGBoost 3-ETF results
│   ├── best_3etf_rotation_results.json   # Best performing rotation
│   └── etf_rotation_comparison.json      # Performance comparison
│
├── 🧠 ML MODELS (H5) - 3-ETF MODELS
│   ├── spy_model.h5                      # SPY prediction model
│   ├── qqq_model.h5                      # QQQ prediction model
│   └── iwm_model.h5                      # IWM prediction model
│
└── 📋 DOCUMENTATION
    ├── FINAL_PROJECT_STRUCTURE.md         # This file
    ├── README_3ETF_STRATEGY.md           # 3-ETF strategy documentation
    ├── ml_analysis_report.txt            # Analysis report
    └── requirements.txt                  # Python dependencies
```

## 🚀 HOW TO USE

### 1. Start the Web Server
```bash
cd ETF-app-3w
node server_clean.js
```
Visit: http://localhost:3023

### 2. Generate 3-ETF Rotation Results
```bash
python fast_ml_strategy_3etf.py
```

### 3. View 3-ETF Analysis
```bash
python ml_result_viewer_fixed.py
```

## 🔗 Key Relationships - 3-ETF ROTATION

1. **fast_ml_strategy_3etf.py** → Analyzes SPY, QQQ, IWM → **JSON result files**
2. **server_clean.js** → Reads 3-ETF JSON files → **Web interface**
3. **etf_data.db** → Contains data for all 3 ETFs → **Both Python & Node.js**

## 📊 3-ETF Web Interface Features

- **ETF Data Overview**: All 3 ETFs (SPY, QQQ, IWM) in rotation view
- **ML Strategy Results**: 3-ETF rotation predictions from all models
- **Trading Performance**: Multi-ETF rotation performance analysis
- **ETF Comparison**: Side-by-side performance metrics

## 🎯 3-ETF ROTATION STRATEGY

**Target ETFs**: 
- **SPY**: S&P 500 ETF (Large Cap)
- **QQQ**: Nasdaq 100 ETF (Tech Heavy)  
- **IWM**: Russell 2000 ETF (Small Cap)

**Strategy**: ML models predict which of the 3 ETFs will perform best in the next period
**Rotation Logic**: Switch to the ETF with highest predicted performance
**Risk Management**: Diversification across market cap sizes

## 🔴 CRITICAL FILES (Never Delete)
- server_clean.js
- etf_data.db  
- fast_ml_strategy_3etf.py
- package.json