const database = require('./database');
const yahooFinance = require('./yahooFinance');
const config = require('./config');
const logger = require('./logger');

class ETFDataFetcher {
    constructor() {
        this.startTime = null;
    }

    async initialize() {
        this.startTime = Date.now();
        logger.info('🚀 Starting ETF data collection system');
        logger.info(`📅 Date range: ${config.dateRange.startDate} to ${config.dateRange.endDate}`);
        logger.info(`📊 ETFs to process: ${config.etfs.length}`);
        
        // Initialize database
        await database.initialize();
        
        // Insert ETF metadata
        await this.insertETFMetadata();
    }

    async insertETFMetadata() {
        logger.info('📝 Inserting ETF metadata into database...');
        
        for (const etf of config.etfs) {
            try {
                await database.insertETF(etf);
                logger.success(`Added ETF: ${etf.symbol} - ${etf.name}`);
            } catch (error) {
                logger.error(`Failed to add ETF ${etf.symbol}:`, error.message);
            }
        }
    }

    async checkExistingData() {
        logger.info('🔍 Checking existing data in database...');
        
        for (const etf of config.etfs) {
            const existing = await database.getExistingDataRange(etf.symbol);
            if (existing.count > 0) {
                logger.info(`${etf.symbol}: ${existing.count} records (${existing.earliest} to ${existing.latest})`);
            } else {
                logger.info(`${etf.symbol}: No existing data`);
            }
        }
    }

    async fetchAllData() {
        logger.info('📈 Starting data fetch from Yahoo Finance...');
        
        const symbols = config.etfs.map(etf => etf.symbol);
        const results = await yahooFinance.fetchMultipleETFs(
            symbols,
            config.dateRange.startDate,
            config.dateRange.endDate
        );
        
        // Process and store results
        let totalRecords = 0;
        let successfulETFs = 0;
        
        for (const [symbol, priceData] of Object.entries(results)) {
            if (priceData.length > 0) {
                try {
                    await database.insertPricesBatch(priceData);
                    totalRecords += priceData.length;
                    successfulETFs++;
                    logger.success(`Stored ${priceData.length} records for ${symbol}`);
                } catch (error) {
                    logger.error(`Failed to store data for ${symbol}:`, error.message);
                }
            } else {
                logger.warning(`No data retrieved for ${symbol}`);
            }
        }
        
        logger.success(`✅ Data collection complete!`);
        logger.info(`📊 Summary:`);
        logger.info(`   - Successful ETFs: ${successfulETFs}/${config.etfs.length}`);
        logger.info(`   - Total price records: ${totalRecords.toLocaleString()}`);
        
        return { successfulETFs, totalRecords };
    }

    async generateReport() {
        const stats = await database.getStats();
        const duration = Math.round((Date.now() - this.startTime)) / 1000;
        
        logger.info('📋 Final Database Report:');
        logger.info(`   - Total ETFs: ${stats.totalETFs}`);
        logger.info(`   - Total price records: ${stats.totalPrices.toLocaleString()}`);
        
        if (stats.dateRange.earliest && stats.dateRange.latest) {
            logger.info(`   - Date range: ${stats.dateRange.earliest} to ${stats.dateRange.latest}`);
            
            // Calculate years of data
            const startYear = new Date(stats.dateRange.earliest).getFullYear();
            const endYear = new Date(stats.dateRange.latest).getFullYear();
            logger.info(`   - Coverage: ${endYear - startYear + 1} years of data`);
        }
        
        logger.info(`   - Processing time: ${duration} seconds`);
        logger.info(`   - Database file: ${config.database.path}`);
        
        // Show sample data for each ETF
        logger.info('🏦 ETF Data Summary:');
        const etfs = await database.getETFList();
        
        for (const etf of etfs) {
            const existing = await database.getExistingDataRange(etf.symbol);
            if (existing.count > 0) {
                logger.info(`   - ${etf.symbol}: ${existing.count.toLocaleString()} records (${existing.earliest} to ${existing.latest})`);
            } else {
                logger.warning(`   - ${etf.symbol}: No data available`);
            }
        }
    }

    async close() {
        database.close();
    }
}

// Main execution function
async function main() {
    const fetcher = new ETFDataFetcher();
    
    try {
        await fetcher.initialize();
        await fetcher.checkExistingData();
        await fetcher.fetchAllData();
        await fetcher.generateReport();
        
        logger.success('🎉 ETF data collection completed successfully!');
        
    } catch (error) {
        logger.error('💥 Fatal error during data collection:', error);
        process.exit(1);
    } finally {
        await fetcher.close();
    }
}

// Export for use as module or run directly
if (require.main === module) {
    main();
}

module.exports = ETFDataFetcher;