const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'etf_data.db');

function listAllData() {
    return new Promise((resolve, reject) => {
        const db = new sqlite3.Database(dbPath, (err) => {
            if (err) {
                console.error('❌ Error opening database:', err);
                reject(err);
                return;
            }
        });

        console.log('\n' + '='.repeat(80));
        console.log('📈 COMPLETE ETF DATABASE - ALL DATA LISTING');
        console.log('='.repeat(80));
        console.log(`📅 Generated: ${new Date().toLocaleString()}`);
        console.log('='.repeat(80));

        // First get database statistics
        db.get(`
            SELECT 
                COUNT(DISTINCT symbol) as etf_count,
                COUNT(*) as total_records,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM prices
        `, (err, stats) => {
            if (err) {
                console.error('❌ Error getting stats:', err);
                return;
            }

            console.log(`🏦 Total ETFs: ${stats.etf_count}`);
            console.log(`💹 Total Price Records: ${stats.total_records}`);
            console.log(`📆 Date Range: ${stats.start_date} to ${stats.end_date}`);
            
            const startYear = new Date(stats.start_date).getFullYear();
            const endYear = new Date(stats.end_date).getFullYear();
            console.log(`⏱️  Coverage: ${endYear - startYear + 1} years`);
            console.log('='.repeat(80));

            // Get all ETFs
            db.all('SELECT * FROM etfs ORDER BY symbol', (err, etfs) => {
                if (err) {
                    console.error('❌ Error getting ETFs:', err);
                    return;
                }

                let etfIndex = 0;
                
                function processNextETF() {
                    if (etfIndex >= etfs.length) {
                        console.log('\n' + '='.repeat(80));
                        console.log('✅ COMPLETE DATA LISTING FINISHED');
                        console.log(`📊 Processed ${etfs.length} ETFs with ${stats.total_records} total records`);
                        console.log('='.repeat(80));
                        db.close();
                        resolve();
                        return;
                    }

                    const etf = etfs[etfIndex];
                    etfIndex++;

                    console.log(`\n${'─'.repeat(60)}`);
                    console.log(`🏷️  ETF ${etfIndex}/${etfs.length}: ${etf.symbol} - ${etf.name}`);
                    console.log(`📂 Sector: ${etf.sector} | Category: ${etf.category}`);
                    console.log(`${'─'.repeat(60)}`);

                    // Get all price data for this ETF
                    db.all(`
                        SELECT date, open, high, low, close, volume, adj_close
                        FROM prices 
                        WHERE symbol = ? 
                        ORDER BY date ASC
                    `, [etf.symbol], (err, prices) => {
                        if (err) {
                            console.error(`❌ Error getting prices for ${etf.symbol}:`, err);
                            processNextETF();
                            return;
                        }

                        if (prices.length === 0) {
                            console.log('❌ No price data available');
                            processNextETF();
                            return;
                        }

                        console.log(`📊 Total Records: ${prices.length}`);
                        console.log(`📅 Date Range: ${prices[0].date} to ${prices[prices.length - 1].date}`);
                        
                        // Show all data for this ETF
                        console.log('\n📈 ALL PRICE RECORDS:');
                        console.log('Date       | Open      | High      | Low       | Close     | Volume      | Adj Close');
                        console.log('-'.repeat(85));
                        
                        for (const price of prices) {
                            console.log(
                                `${price.date} | ` +
                                `$${(price.open || 0).toFixed(2).padStart(8)} | ` +
                                `$${(price.high || 0).toFixed(2).padStart(8)} | ` +
                                `$${(price.low || 0).toFixed(2).padStart(8)} | ` +
                                `$${(price.close || 0).toFixed(2).padStart(8)} | ` +
                                `${(price.volume || 0).toLocaleString().padStart(10)} | ` +
                                `$${(price.adj_close || 0).toFixed(2).padStart(8)}`
                            );
                        }

                        // Show yearly summary
                        console.log('\n📅 YEARLY SUMMARY:');
                        const yearlyData = {};
                        
                        for (const price of prices) {
                            const year = price.date.substring(0, 4);
                            if (!yearlyData[year]) {
                                yearlyData[year] = [];
                            }
                            yearlyData[year].push(price);
                        }

                        console.log('Year | Records | Start Price | End Price   | Min Price   | Max Price   | Return %');
                        console.log('-'.repeat(80));
                        
                        for (const [year, yearPrices] of Object.entries(yearlyData)) {
                            const startPrice = yearPrices[0].close;
                            const endPrice = yearPrices[yearPrices.length - 1].close;
                            const minPrice = Math.min(...yearPrices.map(p => p.low || p.close));
                            const maxPrice = Math.max(...yearPrices.map(p => p.high || p.close));
                            const returnPct = startPrice ? ((endPrice - startPrice) / startPrice * 100) : 0;
                            
                            console.log(
                                `${year} | ${yearPrices.length.toString().padStart(7)} | ` +
                                `$${(startPrice || 0).toFixed(2).padStart(10)} | ` +
                                `$${(endPrice || 0).toFixed(2).padStart(10)} | ` +
                                `$${(minPrice || 0).toFixed(2).padStart(10)} | ` +
                                `$${(maxPrice || 0).toFixed(2).padStart(10)} | ` +
                                `${returnPct.toFixed(1).padStart(7)}%`
                            );
                        }

                        console.log(`\n✅ Completed ${etf.symbol} - ${prices.length} records processed`);
                        
                        // Small delay to prevent overwhelming output
                        setTimeout(processNextETF, 100);
                    });
                }

                processNextETF();
            });
        });
    });
}

// Run the listing
listAllData().catch(console.error);