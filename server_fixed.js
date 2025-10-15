const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3022;

// Serve static files from public directory
app.use(express.static('public'));

// Database connection
const db = new sqlite3.Database('etf_data.db');

// API endpoint to get ETF data
app.get('/api/etfs', (req, res) => {
    db.all("SELECT id, symbol, date, close, sma_5d, monthly_trend FROM prices ORDER BY date DESC LIMIT 100", (err, rows) => {
        if (err) {
            console.log('Database error:', err);
            res.status(500).json({ error: err.message });
            return;
        }
        console.log(`Found ${rows.length} ETF price records`);
        res.json(rows);
    });
});

// API endpoint to get statistics
app.get('/api/stats', (req, res) => {
    db.get(`
        SELECT 
            COUNT(*) as total_records,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            AVG(close) as avg_price
        FROM prices
    `, (err, row) => {
        if (err) {
            console.log('Stats error:', err);
            res.status(500).json({ error: err.message });
            return;
        }
        console.log('Stats retrieved:', row);
        res.json(row);
    });
});

// API endpoint to get ML results
app.get('/api/ml-results', (req, res) => {
    try {
        const results = {};
        const models = ['random_forest', 'xgboost', 'lightgbm'];
        
        console.log('Loading ML results...');
        
        models.forEach(model => {
            const filePath = path.join(__dirname, `${model}_results.json`);
            console.log(`Checking file: ${filePath}`);
            
            if (fs.existsSync(filePath)) {
                try {
                    const fileContent = fs.readFileSync(filePath, 'utf8');
                    results[model] = JSON.parse(fileContent);
                    console.log(`✅ Loaded ${model} results`);
                } catch (parseError) {
                    console.log(`❌ Error parsing ${model} results:`, parseError.message);
                }
            } else {
                console.log(`⚠️ File not found: ${filePath}`);
            }
        });
        
        console.log(`Loaded ${Object.keys(results).length} ML result files`);
        res.json({ success: true, data: results });
    } catch (error) {
        console.log('ML results error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// API endpoint to get trading performance
app.get('/api/trading-performance', (req, res) => {
    try {
        console.log('Loading trading performance...');
        
        // Try multiple possible trading performance files
        const possibleFiles = [
            'trading_performance.json',
            'best_fast_ml_results.json',
            'etf_switching_results.json',
            'improved_strategy_results.json'
        ];
        
        let performanceData = null;
        let foundFile = null;
        
        for (const fileName of possibleFiles) {
            const filePath = path.join(__dirname, fileName);
            console.log(`Checking file: ${filePath}`);
            
            if (fs.existsSync(filePath)) {
                try {
                    const fileContent = fs.readFileSync(filePath, 'utf8');
                    const data = JSON.parse(fileContent);
                    
                    // Check if this file contains trading performance data
                    if (data.backtest_results || data.trades || data.total_return || data.strategy_return) {
                        performanceData = data;
                        foundFile = fileName;
                        console.log(`✅ Found trading performance in ${fileName}`);
                        break;
                    }
                } catch (parseError) {
                    console.log(`❌ Error parsing ${fileName}:`, parseError.message);
                }
            } else {
                console.log(`⚠️ File not found: ${filePath}`);
            }
        }
        

// Main route
=======
        if (performanceData) {
            res.json({ success: true, data: performanceData, source: foundFile });
        } else {
            console.log('❌ No trading performance data found in any file');
            res.json({ success: false, error: 'No trading performance data found' });
        }
    } catch (error) {
        console.log('Trading performance error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Main route
=======

// Main route
app.get('/', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ETF Analysis Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .tabs {
                display: flex;
                background: #333;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                color: white;
                cursor: pointer;
                border: none;
                background: transparent;
                transition: background 0.3s;
            }
            .tab:hover {
                background: #555;
            }
            .tab.active {
                background: #007bff;
            }
            .tab-content {
                padding: 20px;
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .data-table th, .data-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            .data-table th {
                background-color: #f2f2f2;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }
            .stat-label {
                color: #666;
                margin-top: 5px;
            }
            .model-section {
                margin-bottom: 30px;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
            }
            .model-title {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }
            .metric-item {
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .metric-value {
                font-weight: bold;
                color: #007bff;
            }
            .btn {
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin: 5px;
            }
            .btn:hover {
                background: #0056b3;
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            .modal-content {
                background-color: white;
                margin: 5% auto;
                padding: 20px;
                border-radius: 8px;
                width: 80%;
                max-height: 80%;
                overflow-y: auto;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover {
                color: black;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="tabs">
                <button class="tab active" onclick="showTab('etf-data')">ETF Data</button>
                <button class="tab" onclick="showTab('ml-results')">ML Strategy Results</button>
                <button class="tab" onclick="showTab('trading-performance')">Trading Performance</button>
            </div>
            
            <div id="etf-data" class="tab-content active">
                <h2>ETF Data Overview</h2>
                <div id="stats-container">
                    <div class="loading">Loading statistics...</div>
                </div>
                <div id="etf-table-container">
                    <div class="loading">Loading ETF data...</div>
                </div>
            </div>
            
            <div id="ml-results" class="tab-content">
                <h2>Machine Learning Strategy Results</h2>
                <div id="ml-results-container">
                    <div class="loading">Loading ML results...</div>
                </div>
            </div>
            
            <div id="trading-performance" class="tab-content">
                <h2>Trading Performance Analysis</h2>
                <div id="trading-performance-container">
                    <div class="loading">Loading trading performance...</div>
                </div>
            </div>
        </div>

        <!-- Modal for detailed views -->
        <div id="detailModal" class="modal">
            <div class="modal-content">
                <span class="close" onclick="closeModal()">&times;</span>
                <div id="modal-body"></div>
            </div>
        </div>

        <script>
            let mlResultsData = {};
            let tradingPerformanceData = {};

            function showTab(tabName) {
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
                
                // Load data for the selected tab
                if (tabName === 'etf-data') {
                    loadETFData();
                } else if (tabName === 'ml-results') {
                    loadMLResults();
                } else if (tabName === 'trading-performance') {
                    loadTradingPerformance();
                }
            }

            function loadETFData() {
                // Load statistics
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('stats-container').innerHTML = generateStatsHTML(data);
                    })
                    .catch(error => {
                        document.getElementById('stats-container').innerHTML = '<div class="loading">Error loading statistics</div>';
                    });

                // Load ETF data
                fetch('/api/etfs')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('etf-table-container').innerHTML = generateETFTableHTML(data);
                    })
                    .catch(error => {
                        document.getElementById('etf-table-container').innerHTML = '<div class="loading">Error loading ETF data</div>';
                    });
            }

            function loadMLResults() {
                fetch('/api/ml-results')
                    .then(response => response.json())
                    .then(result => {
                        if (result.success) {
                            mlResultsData = result.data;
                            document.getElementById('ml-results-container').innerHTML = generateMLResultsHTML(result.data);
                        } else {
                            document.getElementById('ml-results-container').innerHTML = '<div class="loading">No ML results available</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('ml-results-container').innerHTML = '<div class="loading">Error loading ML results</div>';
                    });
            }

            function loadTradingPerformance() {
                fetch('/api/trading-performance')
                    .then(response => response.json())
                    .then(result => {
                        if (result.success) {
                            tradingPerformanceData = result.data;
                            document.getElementById('trading-performance-container').innerHTML = generateTradingPerformanceHTML(result.data);
                        } else {
                            document.getElementById('trading-performance-container').innerHTML = '<div class="loading">No trading performance data available</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('trading-performance-container').innerHTML = '<div class="loading">Error loading trading performance</div>';
                    });
            }

            function generateStatsHTML(stats) {
                return \`
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">\${stats.total_records}</div>
                            <div class="stat-label">Total Records</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">$\${stats.avg_price ? stats.avg_price.toFixed(2) : 'N/A'}</div>
                            <div class="stat-label">Average Price</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">\${stats.earliest_date || 'N/A'}</div>
                            <div class="stat-label">Earliest Date</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">\${stats.latest_date || 'N/A'}</div>
                            <div class="stat-label">Latest Date</div>
                        </div>
                    </div>
                \`;
            }

            function generateETFTableHTML(data) {
                if (!data || data.length === 0) {
                    return '<div class="loading">No ETF data available</div>';
                }

                let html = '<table class="data-table"><thead><tr>';
                const headers = Object.keys(data[0]);
                headers.forEach(header => {
                    html += \`<th>\${header.replace('_', ' ').toUpperCase()}</th>\`;
                });
                html += '</tr></thead><tbody>';

                data.forEach(row => {
                    html += '<tr>';
                    headers.forEach(header => {
                        let value = row[header];
                        if (typeof value === 'number' && header.includes('price')) {
                            value = '$' + value.toFixed(2);
                        }
                        html += \`<td>\${value || 'N/A'}</td>\`;
                    });
                    html += '</tr>';
                });

                html += '</tbody></table>';
                return html;
            }

            function generateMLResultsHTML(data) {
                if (!data || Object.keys(data).length === 0) {
                    return '<div class="loading">No ML results available</div>';
                }

                let html = '';
                Object.keys(data).forEach(modelName => {
                    const modelData = data[modelName];
                    html += \`
                        <div class="model-section">
                            <div class="model-title">\${modelName.replace('_', ' ').toUpperCase()}</div>
                            <div class="metrics-grid">
                                <div class="metric-item">
                                    <div class="metric-value">\${(modelData.accuracy * 100).toFixed(1)}%</div>
                                    <div>Accuracy</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-value">\${modelData.precision.toFixed(3)}</div>
                                    <div>Precision</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-value">\${modelData.recall.toFixed(3)}</div>
                                    <div>Recall</div>
                                </div>
                                <div class="metric-item">
                                    <div class="metric-value">\${modelData.f1_score.toFixed(3)}</div>
                                    <div>F1 Score</div>
                                </div>
                            </div>
                            <button class="btn" onclick="viewMLDetails('\${modelName}')">View Details</button>
                        </div>
                    \`;
                });

                return html;
            }

            function generateTradingPerformanceHTML(data) {
                if (!data) {
                    return '<div class="loading">No trading performance data available</div>';
                }

                let html = '<div class="stats-grid">';
                
                if (data.summary) {
                    html += \`
                        <div class="stat-card">
                            <div class="stat-value">$\${data.summary.total_return ? data.summary.total_return.toFixed(2) : 'N/A'}</div>
                            <div class="stat-label">Total Return</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">\${data.summary.win_rate ? (data.summary.win_rate * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">\${data.summary.total_trades || 'N/A'}</div>
                            <div class="stat-label">Total Trades</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">\${data.summary.sharpe_ratio ? data.summary.sharpe_ratio.toFixed(2) : 'N/A'}</div>
                            <div class="stat-label">Sharpe Ratio</div>
                        </div>
                    \`;
                }
                
                html += '</div>';

                if (data.models) {
                    Object.keys(data.models).forEach(modelName => {
                        const modelData = data.models[modelName];
                        html += \`
                            <div class="model-section">
                                <div class="model-title">\${modelName.replace('_', ' ').toUpperCase()} Trading Performance</div>
                                <div class="metrics-grid">
                                    <div class="metric-item">
                                        <div class="metric-value">$\${modelData.total_return ? modelData.total_return.toFixed(2) : 'N/A'}</div>
                                        <div>Total Return</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">\${modelData.win_rate ? (modelData.win_rate * 100).toFixed(1) + '%' : 'N/A'}</div>
                                        <div>Win Rate</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">\${modelData.total_trades || 'N/A'}</div>
                                        <div>Total Trades</div>
                                    </div>
                                    <div class="metric-item">
                                        <div class="metric-value">\${modelData.sharpe_ratio ? modelData.sharpe_ratio.toFixed(2) : 'N/A'}</div>
                                        <div>Sharpe Ratio</div>
                                    </div>
                                </div>
                                <button class="btn" onclick="viewTrades('\${modelName}')">View Trades</button>
                            </div>
                        \`;
                    });
                }

                return html;
            }

            function viewMLDetails(modelName) {
                const modelData = mlResultsData[modelName];
                if (!modelData) return;

                let content = \`<h3>\${modelName.replace('_', ' ').toUpperCase()} Details</h3>\`;
                
                if (modelData.feature_importance) {
                    content += '<h4>Feature Importance</h4><ul>';
                    Object.entries(modelData.feature_importance).forEach(([feature, importance]) => {
                        content += \`<li>\${feature}: \${importance.toFixed(4)}</li>\`;
                    });
                    content += '</ul>';
                }

                if (modelData.predictions && modelData.predictions.length > 0) {
                    content += '<h4>Recent Predictions</h4><table class="data-table"><thead><tr><th>Date</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>';
                    modelData.predictions.slice(0, 10).forEach(pred => {
                        content += \`<tr><td>\${pred.date || 'N/A'}</td><td>\${pred.actual || 'N/A'}</td><td>\${pred.predicted || 'N/A'}</td></tr>\`;
                    });
                    content += '</tbody></table>';
                }

                document.getElementById('modal-body').innerHTML = content;
                document.getElementById('detailModal').style.display = 'block';
            }

            function viewTrades(modelName) {
                const modelData = tradingPerformanceData.models && tradingPerformanceData.models[modelName];
                if (!modelData) return;

                let content = \`<h3>\${modelName.replace('_', ' ').toUpperCase()} Trade History</h3>\`;
                
                if (modelData.trades && modelData.trades.length > 0) {
                    content += '<table class="data-table"><thead><tr><th>Date</th><th>Action</th><th>Price</th><th>Quantity</th><th>P&L</th></tr></thead><tbody>';
                    modelData.trades.forEach(trade => {
                        content += \`<tr>
                            <td>\${trade.date || 'N/A'}</td>
                            <td>\${trade.action || 'N/A'}</td>
                            <td>$\${trade.price ? trade.price.toFixed(2) : 'N/A'}</td>
                            <td>\${trade.quantity || 'N/A'}</td>
                            <td class="\${(trade.pnl || 0) >= 0 ? 'positive' : 'negative'}">$\${trade.pnl ? trade.pnl.toFixed(2) : 'N/A'}</td>
                        </tr>\`;
                    });
                    content += '</tbody></table>';
                } else {
                    content += '<p>No trade data available for this model.</p>';
                }

                document.getElementById('modal-body').innerHTML = content;
                document.getElementById('detailModal').style.display = 'block';
            }

            function closeModal() {
                document.getElementById('detailModal').style.display = 'none';
            }

            // Close modal when clicking outside of it
            window.onclick = function(event) {
                const modal = document.getElementById('detailModal');
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            }

            // Load initial data
            loadETFData();
        </script>
    </body>
    </html>
    `);
});

app.listen(port, () => {
    console.log(`ETF Analysis Dashboard running at http://localhost:${port}`);
});