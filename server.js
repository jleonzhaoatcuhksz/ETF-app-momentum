const express = require('express');
const path = require('path');
const database = require('./database');
const logger = require('./logger');

const app = express();
const PORT = process.env.PORT || 3019;

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
        const { startDate, endDate, limit, all } = req.query;
        
        let prices = await database.getPriceData(symbol, startDate, endDate);
        
        // Apply limit if specified (get most recent records)
        if (all !== 'true' && limit) {
            const limitNum = parseInt(limit);
            prices = prices.slice(-limitNum);
        }
        
        res.json({
            success: true,
            symbol: symbol,
            data: prices,
            count: prices.length,
            isComplete: all === 'true',
            dateRange: prices.length > 0 ? {
                start: prices[0].date,
                end: prices[prices.length - 1].date
            } : null
        });
    } catch (error) {
        logger.error(`Error fetching prices for ${req.params.symbol}:`, error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Get detailed ETF analysis
app.get('/api/analysis/:symbol', async (req, res) => {
    try {
        await initializeDatabase();
        const { symbol } = req.params;
        const upperSymbol = symbol.toUpperCase();
        
        // Get ETF info
        const etfs = await database.getETFList();
        const etf = etfs.find(e => e.symbol === upperSymbol);
        
        if (!etf) {
            return res.status(404).json({ 
                success: false, 
                error: 'ETF not found' 
            });
        }
        
        // Get all price data for analysis
        const prices = await database.getPriceData(upperSymbol);
        
        if (prices.length === 0) {
            return res.json({
                success: true,
                etf,
                statistics: null,
                yearly_performance: []
            });
        }
        
        // Calculate statistics
        const closes = prices.map(p => p.close);
        const volumes = prices.map(p => p.volume || 0);
        const firstPrice = prices[0].close;
        const lastPrice = prices[prices.length - 1].close;
        
        const statistics = {
            total_records: prices.length,
            start_date: prices[0].date,
            end_date: prices[prices.length - 1].date,
            min_price: Math.min(...closes),
            max_price: Math.max(...closes),
            avg_price: closes.reduce((sum, price) => sum + price, 0) / closes.length,
            avg_volume: volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length,
            first_price: firstPrice,
            last_price: lastPrice,
            total_return_pct: firstPrice ? ((lastPrice - firstPrice) / firstPrice * 100) : 0
        };
        
        // Calculate yearly performance
        const yearlyData = {};
        
        for (const price of prices) {
            const year = price.date.substring(0, 4);
            if (!yearlyData[year]) {
                yearlyData[year] = [];
            }
            yearlyData[year].push(price);
        }
        
        const yearly_performance = Object.entries(yearlyData).map(([year, yearPrices]) => {
            const closes = yearPrices.map(p => p.close);
            const volumes = yearPrices.map(p => p.volume || 0);
            const yearStart = yearPrices[0].close;
            const yearEnd = yearPrices[yearPrices.length - 1].close;
            
            return {
                year,
                trading_days: yearPrices.length,
                year_low: Math.min(...closes),
                year_high: Math.max(...closes),
                year_start: yearStart,
                year_end: yearEnd,
                avg_volume: volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length,
                return_pct: yearStart ? ((yearEnd - yearStart) / yearStart * 100) : 0
            };
        });
        
        res.json({
            success: true,
            etf,
            statistics,
            yearly_performance
        });
        
    } catch (error) {
        logger.error('Error fetching analysis:', error);
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
        const { symbols, limit = 1 } = req.body;
        
        if (!symbols || !Array.isArray(symbols)) {
            return res.status(400).json({
                success: false,
                error: 'Symbols array is required'
            });
        }
        
        const results = {};
        
        for (const symbol of symbols) {
            try {
                const prices = await database.getPriceData(symbol);
                if (prices.length > 0) {
                    const recentPrices = prices.slice(-parseInt(limit));
                    results[symbol] = {
                        success: true,
                        count: prices.length,
                        latest: recentPrices[recentPrices.length - 1],
                        recent: recentPrices
                    };
                } else {
                    results[symbol] = {
                        success: false,
                        error: 'No price data found'
                    };
                }
            } catch (error) {
                results[symbol] = {
                    success: false,
                    error: error.message
                };
            }
        }
        
        res.json({
            success: true,
            data: results
        });
    } catch (error) {
        logger.error('Error fetching price summary:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        database: dbInitialized ? 'connected' : 'not initialized'
    });
});

// Serve main page
app.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>ETF-app-data</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                .header { text-align: center; margin-bottom: 40px; }
                .stats { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .etf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
                .etf-card { background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; }
                .symbol { font-weight: bold; color: #2196F3; font-size: 18px; }
                button { background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
                button:hover { background: #1976D2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 ETF Data System</h1>
                <p>Historical price data for 14 core ETFs (2015-2025)</p>
            </div>
            
            <div class="stats" id="stats">
                <h3>Database Statistics</h3>
                <p>Loading...</p>
            </div>
            
            <div>
                <h3>Available ETFs</h3>
                <button onclick="loadData()">Load ETF Data</button>
                <div id="etfs" class="etf-grid"></div>
            </div>
            
            <div id="dataPanel" style="display: none; margin-top: 30px; border: 2px solid #2196F3; border-radius: 10px; padding: 20px;">
                <div id="dataPanelContent"></div>
            </div>
            
            <script>
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
                
                async function viewPrices(symbol) {
                    try {
                        const res = await fetch(\`/api/prices/\${symbol}?limit=10\`);
                        const data = await res.json();
                        
                        if (data.data.length === 0) {
                            alert(\`No price data available for \${symbol}\`);
                            return;
                        }
                        
                        let priceInfo = \`Recent prices for \${symbol}:\\n\\n\`;
                        data.data.slice(-5).forEach(price => {
                            priceInfo += \`\${price.date}: $\${price.close.toFixed(2)}\\n\`;
                        });
                        
                        priceInfo += \`\\nTotal records: \${data.count.toLocaleString()}\`;
                        alert(priceInfo);
                        
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
    logger.info(`🌐 ETF Data Web Server starting on port ${PORT}`);
    
    try {
        await initializeDatabase();
        logger.success(`🚀 Server running at http://localhost:${PORT}`);
        logger.info('📊 API Endpoints:');
        logger.info('   GET  /api/etfs - Get all ETFs');
        logger.info('   GET  /api/prices/:symbol - Get price data for ETF');
        logger.info('   GET  /api/stats - Get database statistics');
        logger.info('   POST /api/prices/summary - Get price summary for multiple ETFs');
        logger.info('   GET  /api/health - Health check');
    } catch (error) {
        logger.error('Failed to start server:', error);
        process.exit(1);
    }
});

module.exports = app;