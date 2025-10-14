// Script to calculate and populate Monthly Trend based on 5D-SMA data
const database = require('./database');
const logger = require('./logger');

async function calculateMonthlyTrendForAll() {
    try {
        logger.info('📈 Starting Monthly Trend calculation based on 5D-SMA...');
        
        // Initialize database
        await database.initialize();
        
        // Get all ETFs
        const etfs = await database.getETFList();
        logger.info(`Found ${etfs.length} ETFs to process`);
        
        let totalUpdated = 0;
        
        // Calculate monthly trend for each ETF
        for (const etf of etfs) {
            logger.info(`Processing ${etf.symbol}...`);
            
            const updated = await database.calculateMonthlyTrend(etf.symbol);
            totalUpdated += updated;
            
            logger.success(`${etf.symbol}: Updated ${updated} records with monthly trend`);
        }
        
        logger.success(`🎉 Monthly Trend calculation complete!`);
        logger.info(`Total records updated: ${totalUpdated}`);
        
        // Show sample of updated data
        const sampleData = await new Promise((resolve, reject) => {
            database.db.all(`
                SELECT symbol, date, close, sma_5d, monthly_trend 
                FROM prices 
                WHERE monthly_trend IS NOT NULL 
                ORDER BY symbol, date DESC 
                LIMIT 10
            `, (err, rows) => {
                if (err) reject(err);
                else resolve(rows);
            });
        });
        
        logger.info('\n📊 Sample Monthly Trend Data (based on 5D-SMA):');
        logger.info('Symbol | Date       | Close   | 5D-SMA  | Monthly Trend');
        logger.info('-------|------------|---------|---------|-------------');
        
        sampleData.forEach(row => {
            const trendColor = row.monthly_trend > 0 ? '📈' : row.monthly_trend < 0 ? '📉' : '➡️';
            logger.info(
                `${row.symbol.padEnd(6)} | ${row.date} | $${row.close.toFixed(2).padStart(6)} | $${row.sma_5d.toFixed(2).padStart(6)} | ${trendColor} ${row.monthly_trend.toFixed(2).padStart(6)}%`
            );
        });
        
        // Get statistics
        const stats = await new Promise((resolve, reject) => {
            database.db.get(`
                SELECT 
                    COUNT(*) as total_records,
                    AVG(monthly_trend) as avg_trend,
                    MIN(monthly_trend) as min_trend,
                    MAX(monthly_trend) as max_trend
                FROM prices 
                WHERE monthly_trend IS NOT NULL
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row);
            });
        });
        
        logger.info('\n📈 Monthly Trend Statistics:');
        logger.info(`Total records with monthly trend: ${stats.total_records.toLocaleString()}`);
        logger.info(`Average monthly trend: ${stats.avg_trend.toFixed(2)}%`);
        logger.info(`Range: ${stats.min_trend.toFixed(2)}% to ${stats.max_trend.toFixed(2)}%`);
        
    } catch (error) {
        logger.error('Error calculating monthly trend:', error);
    } finally {
        database.close();
    }
}

// Run if called directly
if (require.main === module) {
    calculateMonthlyTrendForAll();
}

module.exports = calculateMonthlyTrendForAll;