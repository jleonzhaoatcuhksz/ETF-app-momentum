const database = require('./database');
const yahooFinance = require('./yahooFinance');
const config = require('./config');
const logger = require('./logger');

async function runDemo() {
    try {
        logger.info('🎯 Running ETF-app-data demo...');
        
        // Initialize database
        await database.initialize();
        
        // Insert ETF metadata for demo
        logger.info('📝 Setting up demo ETFs...');
        const demoETFs = config.etfs.slice(0, 3); // Just first 3 ETFs for demo
        
        for (const etf of demoETFs) {
            await database.insertETF(etf);
            logger.success(`Added ${etf.symbol}: ${etf.name}`);
        }
        
        // Fetch recent data (last 30 days) for demo
        const endDate = new Date().toISOString().split('T')[0];
        const startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        
        logger.info(`📅 Fetching demo data from ${startDate} to ${endDate}`);
        
        const symbols = demoETFs.map(etf => etf.symbol);
        const results = await yahooFinance.fetchMultipleETFs(symbols, startDate, endDate);
        
        let totalRecords = 0;
        
        // Store demo data
        for (const [symbol, priceData] of Object.entries(results)) {
            if (priceData.length > 0) {
                await database.insertPricesBatch(priceData);
                totalRecords += priceData.length;
                logger.success(`Stored ${priceData.length} records for ${symbol}`);
                
                // Show sample of the data
                const sample = priceData.slice(-3); // Last 3 records
                logger.info(`Sample data for ${symbol}:`);
                sample.forEach(price => {
                    logger.info(`   ${price.date}: $${price.close.toFixed(2)} (Vol: ${price.volume?.toLocaleString() || 'N/A'})`);
                });
            }
        }
        
        // Generate demo report
        const stats = await database.getStats();
        
        logger.success('🎉 Demo completed successfully!');
        logger.info('📊 Demo Results:');
        logger.info(`   - ETFs processed: ${demoETFs.length}`);
        logger.info(`   - Total records: ${totalRecords}`);
        logger.info(`   - Database ETFs: ${stats.totalETFs}`);
        logger.info(`   - Database records: ${stats.totalPrices}`);
        logger.info(`   - Date range: ${stats.dateRange.earliest} to ${stats.dateRange.latest}`);
        
        logger.info('✅ Demo database ready! You can now:');
        logger.info('   - Run "node fetchData.js" to get full historical data');
        logger.info('   - Run "node server.js" to start the web interface');
        
    } catch (error) {
        logger.error('Demo failed:', error);
    } finally {
        database.close();
    }
}

if (require.main === module) {
    runDemo();
}

module.exports = runDemo;