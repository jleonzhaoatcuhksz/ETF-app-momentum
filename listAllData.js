const database = require('./database');
const logger = require('./logger');

class DataLister {
    constructor() {
        this.db = null;
    }

    async initialize() {
        this.db = await database.initialize();
    }

    async listAllETFData() {
        try {
            logger.info('📊 Generating comprehensive ETF data report');
            
            // Get basic statistics
            const stats = await database.getStats();
            console.log('\n' + '='.repeat(80));
            console.log('📈 COMPLETE ETF DATABASE REPORT');
            console.log('='.repeat(80));
            console.log(`📅 Report Generated: ${new Date().toLocaleString()}`);
            console.log(`🏦 Total ETFs: ${stats.totalETFs}`);
            console.log(`💹 Total Price Records: ${stats.totalPrices}`);
            console.log(`📆 Date Range: ${stats.dateRange.start} to ${stats.dateRange.end}`);
            console.log(`⏱️  Coverage: ${stats.coverage} years`);
            console.log('='.repeat(80));

            // Get all ETFs
            const etfs = await database.getAllETFs();
            
            for (const etf of etfs) {
                await this.listETFData(etf);
            }

            console.log('\n' + '='.repeat(80));
            console.log('✅ REPORT COMPLETE - All ETF data listed successfully!');
            console.log('='.repeat(80));

        } catch (error) {
            logger.error('Error generating data report:', error);
        }
    }

    async listETFData(etf) {
        try {
            console.log(`\n${'─'.repeat(60)}`);
            console.log(`🏷️  ETF: ${etf.symbol} - ${etf.name}`);
            console.log(`📂 Sector: ${etf.sector} | Category: ${etf.category}`);
            console.log(`${'─'.repeat(60)}`);

            // Get price data for this ETF
            const query = `
                SELECT date, open, high, low, close, volume, adj_close
                FROM prices 
                WHERE symbol = ? 
                ORDER BY date ASC
            `;
            
            const prices = await new Promise((resolve, reject) => {
                this.db.all(query, [etf.symbol], (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                });
            });

            if (prices.length === 0) {
                console.log('❌ No price data available');
                return;
            }

            console.log(`📊 Total Records: ${prices.length}`);
            console.log(`📅 Date Range: ${prices[0].date} to ${prices[prices.length - 1].date}`);
            
            // Show first 10 records
            console.log('\n📈 FIRST 10 RECORDS:');
            console.log('Date       | Open    | High    | Low     | Close   | Volume    | Adj Close');
            console.log('-'.repeat(75));
            
            for (let i = 0; i < Math.min(10, prices.length); i++) {
                const p = prices[i];
                console.log(
                    `${p.date} | ${p.open?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                    `${p.high?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                    `${p.low?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                    `${p.close?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                    `${p.volume?.toString().padStart(9) || 'N/A'.padStart(9)} | ` +
                    `${p.adj_close?.toFixed(2).padStart(9) || 'N/A'.padStart(9)}`
                );
            }

            // Show last 10 records if more than 10 total
            if (prices.length > 10) {
                console.log('\n📉 LAST 10 RECORDS:');
                console.log('Date       | Open    | High    | Low     | Close   | Volume    | Adj Close');
                console.log('-'.repeat(75));
                
                for (let i = Math.max(0, prices.length - 10); i < prices.length; i++) {
                    const p = prices[i];
                    console.log(
                        `${p.date} | ${p.open?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                        `${p.high?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                        `${p.low?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                        `${p.close?.toFixed(2).padStart(7) || 'N/A'.padStart(7)} | ` +
                        `${p.volume?.toString().padStart(9) || 'N/A'.padStart(9)} | ` +
                        `${p.adj_close?.toFixed(2).padStart(9) || 'N/A'.padStart(9)}`
                    );
                }
            }

            // Show yearly summary
            console.log('\n📅 YEARLY SUMMARY:');
            const yearlyData = this.groupByYear(prices);
            console.log('Year | Records | Start Price | End Price | Min Price | Max Price | Avg Volume');
            console.log('-'.repeat(75));
            
            for (const [year, yearPrices] of Object.entries(yearlyData)) {
                const startPrice = yearPrices[0].close;
                const endPrice = yearPrices[yearPrices.length - 1].close;
                const minPrice = Math.min(...yearPrices.map(p => p.low || p.close));
                const maxPrice = Math.max(...yearPrices.map(p => p.high || p.close));
                const avgVolume = yearPrices.reduce((sum, p) => sum + (p.volume || 0), 0) / yearPrices.length;
                
                console.log(
                    `${year} | ${yearPrices.length.toString().padStart(7)} | ` +
                    `${startPrice?.toFixed(2).padStart(11) || 'N/A'.padStart(11)} | ` +
                    `${endPrice?.toFixed(2).padStart(9) || 'N/A'.padStart(9)} | ` +
                    `${minPrice?.toFixed(2).padStart(9) || 'N/A'.padStart(9)} | ` +
                    `${maxPrice?.toFixed(2).padStart(9) || 'N/A'.padStart(9)} | ` +
                    `${Math.round(avgVolume).toLocaleString().padStart(10)}`
                );
            }

        } catch (error) {
            logger.error(`Error listing data for ${etf.symbol}:`, error);
        }
    }

    groupByYear(prices) {
        const yearlyData = {};
        
        for (const price of prices) {
            const year = price.date.substring(0, 4);
            if (!yearlyData[year]) {
                yearlyData[year] = [];
            }
            yearlyData[year].push(price);
        }
        
        return yearlyData;
    }

    async listRecentData(days = 30) {
        try {
            console.log(`\n📊 RECENT ${days} DAYS DATA FOR ALL ETFs`);
            console.log('='.repeat(80));
            
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - days);
            const cutoffStr = cutoffDate.toISOString().split('T')[0];
            
            const query = `
                SELECT e.symbol, e.name, p.date, p.close, p.volume
                FROM prices p
                JOIN etfs e ON p.symbol = e.symbol
                WHERE p.date >= ?
                ORDER BY p.date DESC, e.symbol ASC
            `;
            
            const recentPrices = await new Promise((resolve, reject) => {
                this.db.all(query, [cutoffStr], (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows);
                });
            });
            
            let currentDate = '';
            let dateCount = 0;
            
            for (const price of recentPrices) {
                if (price.date !== currentDate) {
                    if (currentDate !== '') console.log(''); // Add spacing between dates
                    currentDate = price.date;
                    dateCount++;
                    console.log(`📅 ${currentDate}:`);
                    console.log('-'.repeat(50));
                }
                
                console.log(
                    `  ${price.symbol.padEnd(6)} | $${price.close?.toFixed(2).padStart(8) || 'N/A'.padStart(8)} | ` +
                    `Vol: ${(price.volume || 0).toLocaleString().padStart(12)} | ${price.name}`
                );
                
                // Limit output to prevent overwhelming display
                if (dateCount >= 10) break;
            }
            
        } catch (error) {
            logger.error('Error listing recent data:', error);
        }
    }

    async generateCSVExport() {
        try {
            console.log('\n📄 GENERATING CSV EXPORT SAMPLES...');
            console.log('='.repeat(60));
            
            const etfs = await database.getAllETFs();
            
            for (const etf of etfs) {
                console.log(`\n🏷️  ${etf.symbol} - CSV Format Sample (First 5 rows):`);
                console.log('Date,Symbol,Open,High,Low,Close,Volume,AdjClose,Name');
                
                const query = `
                    SELECT date, open, high, low, close, volume, adj_close
                    FROM prices 
                    WHERE symbol = ? 
                    ORDER BY date ASC
                    LIMIT 5
                `;
                
                const prices = await new Promise((resolve, reject) => {
                    this.db.all(query, [etf.symbol], (err, rows) => {
                        if (err) reject(err);
                        else resolve(rows);
                    });
                });
                
                for (const p of prices) {
                    console.log(
                        `${p.date},${etf.symbol},${p.open || ''},${p.high || ''},${p.low || ''},` +
                        `${p.close || ''},${p.volume || ''},${p.adj_close || ''},"${etf.name}"`
                    );
                }
            }
            
        } catch (error) {
            logger.error('Error generating CSV export:', error);
        }
    }

    async close() {
        if (this.db) {
            this.db.close();
        }
    }
}

async function main() {
    const lister = new DataLister();
    
    try {
        await lister.initialize();
        
        // Generate complete report
        await lister.listAllETFData();
        
        // Show recent data
        await lister.listRecentData(10);
        
        // Generate CSV samples
        await lister.generateCSVExport();
        
    } catch (error) {
        logger.error('Error in main:', error);
    } finally {
        await lister.close();
    }
}

if (require.main === module) {
    main();
}

module.exports = DataLister;