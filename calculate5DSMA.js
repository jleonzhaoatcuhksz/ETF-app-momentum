// Script to calculate and populate 5-day Simple Moving Average
const database = require('./database');
const logger = require('./logger');

async function calculate5DaySMAForAll() {
    try {
        logger.info('🧮 Starting 5-day SMA calculation...');
        
        // Initialize database
        await database.initialize();
        
        // Get all ETFs
        const etfs = await database.getETFList();
        logger.info(`Found ${etfs.length} ETFs to process`);
        
        let totalUpdated = 0;
        
        // Calculate SMA for each ETF
        for (const etf of etfs) {
            logger.info(`Processing ${etf.symbol}...`);
            
            const updated = await database.calculate5DaySMA(etf.symbol);
            totalUpdated += updated;
            
            logger.success(`${etf.symbol}: Updated ${updated} records`);
        }
        
        logger.success(`🎉 5-day SMA calculation complete!`);
        logger.info(`Total records updated: ${totalUpdated}`);
        
        // Show sample of updated data
        const sampleData = await new Promise((resolve, reject) => {
            database.db.all(`
                SELECT symbol, date, close, sma_5d 
                FROM prices 
                WHERE sma_5d IS NOT NULL 
                ORDER BY symbol, date DESC 
                LIMIT 10
            `, (err, rows) => {
                if (err) reject(err);
                else resolve(rows);
            });
        });
        
        logger.info('\n📊 Sample of calculated 5-day SMA data:');
        logger.info('Symbol | Date       | Close   | 5D-SMA');
        logger.info('-------|------------|---------|-------');
        
        sampleData.forEach(row => {
            logger.info(
                `${row.symbol.padEnd(6)} | ${row.date} | $${row.close.toFixed(2).padStart(6)} | $${row.sma_5d.toFixed(2).padStart(6)}`
            );
        });
        
    } catch (error) {
        logger.error('Error calculating 5-day SMA:', error);
    } finally {
        database.close();
    }
}

// Run if called directly
if (require.main === module) {
    calculate5DaySMAForAll();
}

module.exports = calculate5DaySMAForAll;