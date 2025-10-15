// ETF Data Configuration
module.exports = {
    // Database configuration
    database: {
        path: './etf_data.db',
        resetOnStart: false
    },

    // Date range for historical data (over 10 years)
    dateRange: {
        startDate: '2015-10-10',
        endDate: '2025-10-10'
    },

    // Single ETF - No Switching Strategy (1 total)
    etfs: [
        // Focus on SPY only - Single ETF Buy & Hold Strategy
        {
            symbol: 'SPY',
            name: 'SPDR S&P 500 ETF Trust',
            sector: 'Large Cap Blend',
            category: 'Single ETF - No Switching'
        }
    ],

    // Yahoo Finance API settings
    yahooFinance: {
        baseUrl: 'https://query1.finance.yahoo.com/v8/finance/chart',
        interval: '1d',
        retryDelay: 1000,
        maxRetries: 3,
        requestDelay: 300, // 300ms delay between requests
        batchSize: 3 // Process 3 ETFs at a time to avoid rate limiting
    },

    // Logging configuration
    logging: {
        level: 'info',
        enableConsole: true,
        enableFile: false
    }
};