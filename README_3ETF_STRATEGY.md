# 3-ETF Rotation Strategy

## 🎯 Strategy Overview

The 3-ETF Rotation Strategy uses machine learning to dynamically rotate between three major ETFs:

- **SPY**: S&P 500 ETF (Large Cap Stocks)
- **QQQ**: Nasdaq 100 ETF (Technology Heavy)  
- **IWM**: Russell 2000 ETF (Small Cap Stocks)

## 🧠 Machine Learning Approach

### Models Used
1. **Random Forest**: Ensemble method for robust predictions
2. **XGBoost**: Gradient boosting for high performance
3. **LightGBM**: Fast gradient boosting with efficiency

### Features
- **Price Returns**: 1-day, 5-day, 20-day returns
- **Technical Indicators**: SMA ratios, volatility
- **Relative Performance**: Each ETF vs others
- **Market Trends**: Monthly trend indicators

### Prediction Target
The model predicts which ETF will be the **best performer** in the next trading period.

## 📊 Strategy Logic

1. **Data Analysis**: Analyze historical performance of all 3 ETFs
2. **Feature Engineering**: Create predictive features from price and volume data
3. **Model Training**: Train ML models to predict best-performing ETF
4. **Rotation Decision**: Switch to ETF with highest predicted performance
5. **Risk Management**: Diversify across different market segments

## 🚀 Expected Benefits

### Diversification
- **Large Cap** (SPY): Stable, dividend-paying companies
- **Tech Focus** (QQQ): Growth-oriented technology stocks
- **Small Cap** (IWM): Higher growth potential, higher volatility

### Dynamic Allocation
- Automatically switches between market segments
- Captures momentum across different market conditions
- Reduces single-ETF concentration risk

### ML-Driven Decisions
- Data-driven rotation decisions
- Removes emotional bias from trading
- Adapts to changing market conditions

## 📈 Performance Metrics

The system tracks:
- **Strategy Return**: Total return of rotation strategy
- **Benchmark Return**: SPY buy-and-hold performance
- **Outperformance**: Strategy vs benchmark
- **Number of Rotations**: Trading frequency
- **Model Accuracy**: ML prediction accuracy

## 🔧 Technical Implementation

### Data Pipeline
```
ETF Price Data → Feature Engineering → ML Model Training → Predictions → Trading Signals
```

### Web Interface
- Real-time ETF performance comparison
- ML model predictions and confidence scores
- Trading history and performance analytics
- Interactive charts and visualizations

## ⚠️ Risk Considerations

1. **Model Risk**: ML predictions may not always be accurate
2. **Transaction Costs**: Frequent rotations incur trading fees
3. **Market Risk**: All ETFs subject to overall market conditions
4. **Overfitting Risk**: Models may not generalize to future data

## 🎯 Usage Instructions

1. **Run Analysis**: `python fast_ml_strategy_3etf.py`
2. **Start Web Server**: `node server_clean.js`
3. **View Results**: http://localhost:3023
4. **Monitor Performance**: Check trading results and model accuracy

## 📋 Files Structure

- `fast_ml_strategy_3etf.py`: Main strategy implementation
- `server_clean.js`: Web interface server
- `3etf_rotation_results.json`: Strategy results
- `etf_data.db`: Historical ETF data