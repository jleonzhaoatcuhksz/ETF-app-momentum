const axios = require('axios');
const config = require('./config');
const logger = require('./logger');

class YahooFinanceClient {
    constructor() {
        this.baseUrl = config.yahooFinance.baseUrl;
        this.requestDelay = config.yahooFinance.requestDelay;
        this.maxRetries = config.yahooFinance.maxRetries;
        this.retryDelay = config.yahooFinance.retryDelay;
        
        // Create axios instance with timeout
        this.client = axios.create({
            timeout: 30000,
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        });
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    convertTimestampToDate(timestamp) {
        return new Date(timestamp * 1000).toISOString().split('T')[0];
    }

    async fetchPriceData(symbol, startDate, endDate) {
        const startTimestamp = Math.floor(new Date(startDate).getTime() / 1000);
        const endTimestamp = Math.floor(new Date(endDate).getTime() / 1000);
        
        const url = `${this.baseUrl}/${symbol}`;
        const params = {
            period1: startTimestamp,
            period2: endTimestamp,
            interval: config.yahooFinance.interval,
            includePrePost: false,
            events: 'div,splits'
        };

        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                logger.info(`Fetching ${symbol} data (attempt ${attempt}/${this.maxRetries})`);
                
                const response = await this.client.get(url, { params });
                
                if (!response.data || !response.data.chart || !response.data.chart.result) {
                    throw new Error('Invalid response structure from Yahoo Finance');
                }

                const result = response.data.chart.result[0];
                
                if (!result.timestamp || !result.indicators.quote[0]) {
                    logger.warning(`No data available for ${symbol} in specified date range`);
                    return [];
                }

                const timestamps = result.timestamp;
                const quotes = result.indicators.quote[0];
                const adjClose = result.indicators.adjclose ? result.indicators.adjclose[0].adjclose : null;

                const priceData = [];
                
                for (let i = 0; i < timestamps.length; i++) {
                    // Skip entries with null closing prices
                    if (quotes.close[i] === null || quotes.close[i] === undefined) {
                        continue;
                    }

                    priceData.push({
                        symbol: symbol,
                        date: this.convertTimestampToDate(timestamps[i]),
                        open: quotes.open[i],
                        high: quotes.high[i],
                        low: quotes.low[i],
                        close: quotes.close[i],
                        volume: quotes.volume[i],
                        adj_close: adjClose ? adjClose[i] : quotes.close[i]
                    });
                }

                logger.success(`Successfully fetched ${priceData.length} price records for ${symbol}`);
                
                // Add delay between requests to avoid rate limiting
                if (this.requestDelay > 0) {
                    await this.delay(this.requestDelay);
                }
                
                return priceData;

            } catch (error) {
                logger.warning(`Attempt ${attempt} failed for ${symbol}: ${error.message}`);
                
                if (attempt === this.maxRetries) {
                    logger.error(`Failed to fetch data for ${symbol} after ${this.maxRetries} attempts`);
                    throw error;
                }
                
                // Wait before retry
                await this.delay(this.retryDelay * attempt);
            }
        }
    }

    async fetchMultipleETFs(symbols, startDate, endDate) {
        const results = {};
        const batchSize = config.yahooFinance.batchSize;
        
        logger.info(`Fetching data for ${symbols.length} ETFs in batches of ${batchSize}`);
        
        for (let i = 0; i < symbols.length; i += batchSize) {
            const batch = symbols.slice(i, i + batchSize);
            logger.info(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(symbols.length/batchSize)}: ${batch.join(', ')}`);
            
            const batchPromises = batch.map(symbol => 
                this.fetchPriceData(symbol, startDate, endDate)
                    .then(data => ({ symbol, data, success: true }))
                    .catch(error => ({ symbol, error: error.message, success: false }))
            );
            
            const batchResults = await Promise.all(batchPromises);
            
            batchResults.forEach(result => {
                if (result.success) {
                    results[result.symbol] = result.data;
                } else {
                    logger.error(`Failed to fetch ${result.symbol}: ${result.error}`);
                    results[result.symbol] = [];
                }
            });
            
            // Longer delay between batches
            if (i + batchSize < symbols.length) {
                logger.info(`Waiting before next batch...`);
                await this.delay(this.requestDelay * 3);
            }
        }
        
        return results;
    }
}

module.exports = new YahooFinanceClient();