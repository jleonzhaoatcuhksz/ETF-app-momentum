const express = require('express');
const path = require('path');
const database = require('./database');
const logger = require('./logger');
const momentum = require('./momentum');

const app = express();
const PORT = process.env.PORT || 3021;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Initialize database on server start
let dbInitialized = false;

async function initializeDatabase() {
    if (!dbInitialized) {
        try {
            await database.initialize();
            dbInitialized = true;
            logger.success('Database initialized for web server');
        } catch (error) {
            logger.error('Failed to initialize database:', error);
            throw error;
        }
    }
}

// API Routes

// Get ML Results
app.get('/api/ml-results', async (req, res) => {
    try {
        const fs = require('fs');
        const results = {};
        
        // Load all ML result files
        const resultFiles = [
            'best_fast_ml_results.json',
            'random_forest_results.json', 
            'xgboost_results.json',
            'lightgbm_results.json',
            'improved_strategy_results.json',
            'etf_switching_results.json'
        ];
        
        for (const file of resultFiles) {
            if (fs.existsSync(file)) {
                try {
                    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
                    const modelName = file.replace('_results.json', '').replace('.json', '');
                    results[modelName] = data;
                } catch (e) {
                    logger.error(`Error loading ${file}:`, e);
                }
            }
        }
        
        res.json({
            success: true,
            data: results,
            count: Object.keys(results).length
        });
    } catch (error) {
        logger.error('Error fetching ML results:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get trading performance summary
app.get('/api/trading-performance', async (req, res) => {
    try {
        const fs = require('fs');
        const performance = [];
        
        const resultFiles = [
            'best_fast_ml_results.json',
            'random_forest_results.json', 
            'xgboost_results.json',
            'lightgbm_results.json'
        ];
        
        for (const file of resultFiles) {
            if (fs.existsSync(file)) {
                try {
                    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
                    const modelName = file.replace('_results.json', '').replace('.json', '').toUpperCase();
                    
                    performance.push({
                        model: modelName,
                        strategy_return: data.strategy_return || 0,
                        spy_return: data.spy_return || 0,
                        outperformance: data.outperformance || 0,
                        total_trades: data.total_trades || 0,
                        final_portfolio_value: data.final_portfolio_value || 0,
                        trades: data.trades || []
                    });
                } catch (e) {
                    logger.error(`Error loading ${file}:`, e);
                }
            }
        }
        
        // Sort by outperformance
        performance.sort((a, b) => b.outperformance - a.outperformance);
        
        res.json({
            success: true,
            data: performance,
            count: performance.length
        });
    } catch (error) {
        logger.error('Error fetching trading performance:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get all ETFs
app.get('/api/etfs', async (req, res) => {
    try {
        await initializeDatabase();
        const etfs = await database.getETFList();
        res.json({
            success: true,
            data: etfs,
            count: etfs.length
        });
    } catch (error) {
        logger.error('Error fetching ETFs:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get price data for specific ETF
app.get('/api/prices/:symbol', async (req, res) => {
    try {
        await initializeDatabase();
        const { symbol } = req.params;
        const { limit = 100, offset = 0 } = req.query;
        
        const prices = await database.getPriceData(symbol, parseInt(limit), parseInt(offset));
        const count = await database.getPriceCount(symbol);
        
        res.json({
            success: true,
            data: prices,
            count: count,
            symbol: symbol
        });
    } catch (error) {
        logger.error(`Error fetching prices for ${req.params.symbol}:`, error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get database statistics
app.get('/api/stats', async (req, res) => {
    try {
        await initializeDatabase();
        const stats = await database.getStats();
        res.json({
            success: true,
            data: stats
        });
    } catch (error) {
        logger.error('Error fetching stats:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get price summary for multiple ETFs
app.post('/api/prices/summary', async (req, res) => {
    try {
        await initializeDatabase();
        const { symbols, startDate, endDate } = req.body;
        
        if (!symbols || !Array.isArray(symbols)) {
            return res.status(400).json({
                success: false,
                error: 'symbols array is required'
            });
        }
        
        const summary = {};
        
        for (const symbol of symbols) {
            try {
                const prices = await database.getPriceDataByDateRange(symbol, startDate, endDate);
                summary[symbol] = {
                    count: prices.length,
                    data: prices.slice(-10) // Last 10 records
                };
            } catch (err) {
                summary[symbol] = {
                    count: 0,
                    data: [],
                    error: err.message
                };
            }
        }
        
        res.json({
            success: true,
            data: summary
        });
    } catch (error) {
        logger.error('Error fetching price summary:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        database: dbInitialized ? 'connected' : 'not connected'
    });
});

// Serve main page
app.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>ETF Data & ML Analysis Viewer</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }
                .nav-tabs {
                    display: flex;
                    background: #34495e;
                    border-radius: 8px 8px 0 0;
                    margin: -30px -30px 20px -30px;
                }
                .nav-tab {
                    flex: 1;
                    padding: 15px;
                    text-align: center;
                    background: #34495e;
                    color: white;
                    cursor: pointer;
                    border: none;
                    transition: background 0.3s;
                }
                .nav-tab:first-child { border-radius: 8px 0 0 0; }
                .nav-tab:last-child { border-radius: 0 8px 0 0; }
                .nav-tab.active { background: #3498db; }
                .nav-tab:hover { background: #2980b9; }
                
                .tab-content { display: none; }
                .tab-content.active { display: block; }
                
                h1 { 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .etf-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 15px; 
                    margin-top: 20px;
                }
                .etf-card, .ml-card { 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    border-radius: 8px; 
                    background: #f9f9f9;
                    transition: transform 0.2s;
                }
                .etf-card:hover, .ml-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }
                .symbol { 
                    font-weight: bold; 
                    font-size: 18px; 
                    color: #2980b9; 
                    margin-bottom: 5px;
                }
                .ml-model {
                    font-weight: bold; 
                    font-size: 18px; 
                    color: #27ae60; 
                    margin-bottom: 5px;
                }
                button {
                    background: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    margin: 2px;
                }
                button:hover { background: #2980b9; }
                button.success { background: #27ae60; }
                button.success:hover { background: #229954; }
                
                #stats {
                    background: #e8f6f3;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #27ae60;
                }
                .performance-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                .performance-table th, .performance-table td {
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }
                .performance-table th {
                    background: #3498db;
                    color: white;
                }
                .performance-table tr:nth-child(even) {
                    background: #f2f2f2;
                }
                .success-row { background: #d5f4e6 !important; }
                .champion { background: #f1c40f !important; color: #2c3e50; font-weight: bold; }
                
                .trade-details {
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 10px 0;
                    font-family: monospace;
                    font-size: 12px;
                    max-height: 300px;
                    overflow-y: auto;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="nav-tabs">
                    <button class="nav-tab active" onclick="showTab('etf-data')">📊 ETF Data</button>
                    <button class="nav-tab" onclick="showTab('ml-results')">🤖 ML Results</button>
                    <button class="nav-tab" onclick="showTab('trading-performance')">📈 Trading Performance</button>
                </div>
                
                <!-- ETF Data Tab -->
                <div id="etf-data" class="tab-content active">
                    <h1>📊 ETF Data Viewer</h1>
                    
                    <div id="stats">
                        <h3>Loading statistics...</h3>
                    </div>
                    
                    <div id="etfs" class="etf-grid">
                        <p>Loading ETF data...</p>
                    </div>
                </div>
                
                <!-- ML Results Tab -->
                <div id="ml-results" class="tab-content">
                    <h1>🤖 ML Strategy Results</h1>
                    <div id="ml-content">
                        <p>Loading ML results...</p>
                    </div>
                </div>
                
                <!-- Trading Performance Tab -->
                <div id="trading-performance" class="tab-content">
                    <h1>📈 Trading Performance Analysis</h1>
                    <div id="performance-content">
                        <p>Loading trading performance...</p>
                    </div>
                </div>
            </div>
            
            <div id="dataPanel" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000;">
                <div style="background: white; margin: 50px auto; padding: 20px; width: 90%; max-height: 80%; overflow-y: auto; border-radius: 10px;">
                    <button onclick="document.getElementById('dataPanel').style.display='none'" style="float: right;">✕ Close</button>
                    <div id="dataPanelContent"></div>
                </div>
            </div>
            
            <script>
                // Tab management
                function showTab(tabName) {
                    // Hide all tabs
                    document.querySelectorAll('.tab-content').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    document.querySelectorAll('.nav-tab').forEach(navTab => {
                        navTab.classList.remove('active');
                    });
                    
                    // Show selected tab
                    document.getElementById(tabName).classList.add('active');
                    event.target.classList.add('active');
                    
                    // Load data for specific tabs
                    if (tabName === 'ml-results') {
                        loadMLResults();
                    } else if (tabName === 'trading-performance') {
                        loadTradingPerformance();
                    }
                }
                
                async function loadData() {
                    try {
                        // Load stats
                        const statsRes = await fetch('/api/stats');
                        const stats = await statsRes.json();
                        
                        document.getElementById('stats').innerHTML = \`
                            <h3>📈 Database Statistics</h3>
                            <p><strong>Total ETFs:</strong> \${stats.data.totalETFs}</p>
                            <p><strong>Total Price Records:</strong> \${stats.data.totalPrices.toLocaleString()}</p>
                            <p><strong>Date Range:</strong> \${stats.data.dateRange.earliest || 'N/A'} to \${stats.data.dateRange.latest || 'N/A'}</p>
                        \`;
                        
                        // Load ETFs
                        const etfsRes = await fetch('/api/etfs');
                        const etfs = await etfsRes.json();
                        
                        let html = '';
                        etfs.data.forEach(etf => {
                            html += \`
                                <div class="etf-card">
                                    <div class="symbol">\${etf.symbol}</div>
                                    <div>\${etf.name}</div>
                                    <div><strong>Sector:</strong> \${etf.sector}</div>
                                    <div><strong>Category:</strong> \${etf.category}</div>
                                    <button onclick="viewPrices('\${etf.symbol}')" style="margin-top: 10px;">View Prices</button>
                                </div>
                            \`;
                        });
                        
                        document.getElementById('etfs').innerHTML = html;
                        
                    } catch (error) {
                        console.error('Error loading data:', error);
                        alert('Error loading data. Make sure the database is populated.');
                    }
                }
                
                async function loadMLResults() {
                    try {
                        const res = await fetch('/api/ml-results');
                        const data = await res.json();
                        
                        let html = '<div class="etf-grid">';
                        
                        Object.entries(data.data).forEach(([modelName, results]) => {
                            const outperformance = results.outperformance || 0;
                            const isSuccess = outperformance > 0;
                            
                            html += \`
                                <div class="ml-card \${isSuccess ? 'success-row' : ''}">
                                    <div class="ml-model">\${modelName.toUpperCase()}</div>
                                    <div><strong>Strategy Return:</strong> \${(results.strategy_return || 0).toFixed(2)}%</div>
                                    <div><strong>SPY Benchmark:</strong> \${(results.spy_return || 0).toFixed(2)}%</div>
                                    <div><strong>Outperformance:</strong> <span style="color: \${isSuccess ? '#27ae60' : '#e74c3c'}">\${outperformance.toFixed(2)}%</span></div>
                                    <div><strong>Total Trades:</strong> \${results.total_trades || 0}</div>
                                    <div><strong>Final Portfolio:</strong> $\${(results.final_portfolio_value || 0).toLocaleString()}</div>
                                    <button class="\${isSuccess ? 'success' : ''}" onclick="viewMLDetails('\${modelName}', '\${btoa(JSON.stringify(results))}')">\${isSuccess ? '🏆 View Success' : '📊 View Details'}</button>
                                </div>
                            \`;
                        });
                        
                        html += '</div>';
                        
                        if (Object.keys(data.data).length === 0) {
                            html = '<p>No ML results found. Run the ML strategies first.</p>';
                        }
                        
                        document.getElementById('ml-content').innerHTML = html;
                        
                    } catch (error) {
                        console.error('Error loading ML results:', error);
                        document.getElementById('ml-content').innerHTML = '<p>Error loading ML results.</p>';
                    }
                }
                
                async function loadTradingPerformance() {
                    try {
                        const res = await fetch('/api/trading-performance');
                        const data = await res.json();
                        
                        if (data.data.length === 0) {
                            document.getElementById('performance-content').innerHTML = '<p>No trading performance data found.</p>';
                            return;
                        }
                        
                        let html = \`
                            <div style="background: #f1c40f; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #2c3e50;">
                                <h3>🏆 CHAMPION: \${data.data[0].model}</h3>
                                <p><strong>Best Outperformance:</strong> +\${data.data[0].outperformance.toFixed(2)}% vs SPY</p>
                            </div>
                            
                            <table class="performance-table">
                                <tr>
                                    <th>Model</th>
                                    <th>Strategy Return</th>
                                    <th>SPY Return</th>
                                    <th>Outperformance</th>
                                    <th>Total Trades</th>
                                    <th>Final Portfolio</th>
                                    <th>Action</th>
                                </tr>
                        \`;
                        
                        data.data.forEach((perf, index) => {
                            const isChampion = index === 0;
                            const isSuccess = perf.outperformance > 0;
                            
                            html += \`
                                <tr class="\${isChampion ? 'champion' : isSuccess ? 'success-row' : ''}">
                                    <td><strong>\${perf.model}</strong></td>
                                    <td>+\${perf.strategy_return.toFixed(2)}%</td>
                                    <td>+\${perf.spy_return.toFixed(2)}%</td>
                                    <td style="color: \${isSuccess ? '#27ae60' : '#e74c3c'}">+\${perf.outperformance.toFixed(2)}%</td>
                                    <td>\${perf.total_trades}</td>
                                    <td>$\${perf.final_portfolio_value.toLocaleString()}</td>
                                    <td><button onclick="viewTrades('\${perf.model}', \${btoa(JSON.stringify(perf.trades))})">View Trades</button></td>
                                </tr>
                            \`;
                        });
                        
                        html += '</table>';
                        
                        document.getElementById('performance-content').innerHTML = html;
                        
                    } catch (error) {
                        console.error('Error loading trading performance:', error);
                        document.getElementById('performance-content').innerHTML = '<p>Error loading trading performance.</p>';
                    }
                }
                
                function viewMLDetails(modelName, results) {
                    let content = \`
                        <h2>🤖 \${modelName.toUpperCase()} - Detailed Results</h2>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                            <div>
                                <h3>📊 Performance Metrics</h3>
                                <p><strong>Strategy Return:</strong> \${results.strategy_return.toFixed(2)}%</p>
                                <p><strong>SPY Benchmark:</strong> \${results.spy_return.toFixed(2)}%</p>
                                <p><strong>Outperformance:</strong> <span style="color: \${results.outperformance > 0 ? '#27ae60' : '#e74c3c'}; font-weight: bold;">\${results.outperformance.toFixed(2)}%</span></p>
                                <p><strong>Final Portfolio Value:</strong> $\${results.final_portfolio_value.toLocaleString()}</p>
                                <p><strong>Final SPY Value:</strong> $\${results.final_spy_value.toLocaleString()}</p>
                            </div>
                            <div>
                                <h3>💼 Trading Summary</h3>
                                <p><strong>Total Trades:</strong> \${results.total_trades}</p>
                                <p><strong>Model Type:</strong> \${results.model_name || modelName}</p>
                                <p><strong>Status:</strong> <span style="color: \${results.outperformance > 0 ? '#27ae60' : '#e74c3c'}; font-weight: bold;">\${results.outperformance > 0 ? '🏆 SUCCESS' : '⚠️ UNDERPERFORMED'}</span></p>
                            </div>
                        </div>
                    \`;
                    
                    if (results.trades && results.trades.length > 0) {
                        content += '<h3>📋 Trade Details</h3><div class="trade-details">';
                        results.trades.forEach((trade, index) => {
                            content += \`Trade \${index + 1}: \${trade.date} - \${trade.from_etf || 'N/A'} → \${trade.to_etf || 'N/A'} (Portfolio: $\${(trade.portfolio_value || 0).toLocaleString()})\\n\`;
                        });
                        content += '</div>';
                    }
                    
                    document.getElementById('dataPanelContent').innerHTML = content;
                    document.getElementById('dataPanel').style.display = 'block';
                }
                
                function viewTrades(modelName, trades) {
                    let content = \`<h2>📈 \${modelName} - Trading History</h2>\`;
                    
                    if (trades && trades.length > 0) {
                        content += '<div class="trade-details">';
                        trades.forEach((trade, index) => {
                            const portfolioValue = trade.portfolio_value || 0;
                            content += \`Trade \${index + 1}: \${trade.date || 'N/A'} - \${trade.from_etf || 'N/A'} → \${trade.to_etf || 'N/A'} (Portfolio: $\${portfolioValue.toLocaleString()})\\n\`;
                        });
                        content += '</div>';
                    } else {
                        content += '<p>No trading history available.</p>';
                    }
                    
                    document.getElementById('dataPanelContent').innerHTML = content;
                    document.getElementById('dataPanel').style.display = 'block';
                }
                
                async function viewPrices(symbol) {
                    try {
                        const res = await fetch(\`/api/prices/\${symbol}?limit=10\`);
                        const data = await res.json();
                        
                        if (data.data.length === 0) {
                            alert(\`No price data available for \${symbol}\`);
                            return;
                        }
                        
                        let content = \`<h2>📊 \${symbol} - Recent Prices</h2><div class="trade-details">\`;
                        data.data.slice(-10).forEach(price => {
                            content += \`\${price.date}: $\${price.close.toFixed(2)}\\n\`;
                        });
                        content += \`\\nTotal records: \${data.count.toLocaleString()}</div>\`;
                        
                        document.getElementById('dataPanelContent').innerHTML = content;
                        document.getElementById('dataPanel').style.display = 'block';
                        
                    } catch (error) {
                        console.error('Error loading prices:', error);
                        alert('Error loading price data');
                    }
                }
                
                // Auto-load data when page loads
                loadData();
            </script>
        </body>
        </html>
    `);
});

// Start server
app.listen(PORT, async () => {
    logger.info(`🌐 ETF Data & ML Analysis Web Server starting on port ${PORT}`);
    
    try {
        await initializeDatabase();
        logger.success(`🚀 Server running at http://localhost:${PORT}`);
        logger.info('📊 API Endpoints:');
        logger.info('   GET  /api/etfs - Get all ETFs');
        logger.info('   GET  /api/prices/:symbol - Get price data for ETF');
        logger.info('   GET  /api/stats - Get database statistics');
        logger.info('   GET  /api/ml-results - Get ML strategy results');
        logger.info('   GET  /api/trading-performance - Get trading performance data');
        logger.info('   POST /api/prices/summary - Get price summary for multiple ETFs');
        logger.info('   GET  /api/health - Health check');
    } catch (error) {
        logger.error('Failed to start server:', error);
        process.exit(1);
    }
});

module.exports = app;