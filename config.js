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

    // Core ETF symbols (14 total)
    etfs: [
        // Core ETFs (3)
        {
            symbol: 'SPY',
            name: 'SPDR S&P 500 ETF Trust',
            sector: 'Large Cap Blend',
            category: 'Core ETF'
        },
        {
            symbol: 'TLT',
            name: 'iShares 20+ Year Treasury Bond ETF',
            sector: 'Long-Term Treasury',
            category: 'Core ETF'
        },
        {
            symbol: 'SHY',
            name: 'iShares 1-3 Year Treasury Bond ETF',
            sector: 'Short-Term Treasury',
            category: 'Core ETF'
        },

        // iShares Sector ETFs (11 main sectors)
        {
            symbol: 'XLK',
            name: 'Technology Select Sector SPDR Fund',
            sector: 'Technology',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLF',
            name: 'Financial Select Sector SPDR Fund',
            sector: 'Financials',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLV',
            name: 'Health Care Select Sector SPDR Fund',
            sector: 'Health Care',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLI',
            name: 'Industrial Select Sector SPDR Fund',
            sector: 'Industrials',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLE',
            name: 'Energy Select Sector SPDR Fund',
            sector: 'Energy',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLY',
            name: 'Consumer Discretionary Select Sector SPDR Fund',
            sector: 'Consumer Discretionary',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLP',
            name: 'Consumer Staples Select Sector SPDR Fund',
            sector: 'Consumer Staples',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLU',
            name: 'Utilities Select Sector SPDR Fund',
            sector: 'Utilities',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLB',
            name: 'Materials Select Sector SPDR Fund',
            sector: 'Materials',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLRE',
            name: 'Real Estate Select Sector SPDR Fund',
            sector: 'Real Estate',
            category: 'Sector ETF'
        },
        {
            symbol: 'XLC',
            name: 'Communication Services Select Sector SPDR Fund',
            sector: 'Communication Services',
            category: 'Sector ETF'
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