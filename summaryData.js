const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, 'etf_data.db');

function showDataSummary() {
    return new Promise((resolve, reject) => {
        const db = new sqlite3.Database(dbPath, (err) => {
            if (err) {
                console.error('❌ Error opening database:', err);
                reject(err);
                return;
            }
        });

        console.log('\n' + '='.repeat(80));
        console.log('📈 ETF DATABASE - 10 YEARS DATA SUMMARY');
        console.log('='.repeat(80));
        console.log(`📅 Generated: ${new Date().toLocaleString()}`);
        console.log('='.repeat(80));

        // Get database statistics
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

            // Get ETF summary with record counts
            db.all(`
                SELECT 
                    e.symbol,
                    e.name,
                    e.sector,
                    COUNT(p.date) as record_count,
                    MIN(p.date) as start_date,
                    MAX(p.date) as end_date,
                    MIN(p.close) as min_price,
                    MAX(p.close) as max_price,
                    (SELECT close FROM prices WHERE symbol = e.symbol ORDER BY date ASC LIMIT 1) as first_price,
                    (SELECT close FROM prices WHERE symbol = e.symbol ORDER BY date DESC LIMIT 1) as last_price
                FROM etfs e
                LEFT JOIN prices p ON e.symbol = p.symbol
                GROUP BY e.symbol
                ORDER BY e.symbol
            `, (err, etfSummary) => {
                if (err) {
                    console.error('❌ Error getting ETF summary:', err);
                    return;
                }

                console.log('\n📊 ETF OVERVIEW:');
                console.log('Symbol | Name                              | Sector        | Records | Date Range        | Price Range      | Total Return');
                console.log('-'.repeat(130));
                
                for (const etf of etfSummary) {
                    const totalReturn = etf.first_price ? ((etf.last_price - etf.first_price) / etf.first_price * 100) : 0;
                    console.log(
                        `${etf.symbol.padEnd(6)} | ` +
                        `${etf.name.substring(0, 33).padEnd(33)} | ` +
                        `${etf.sector.substring(0, 13).padEnd(13)} | ` +
                        `${etf.record_count.toString().padStart(7)} | ` +
                        `${etf.start_date} to ${etf.end_date.substring(5)} | ` +
                        `$${etf.min_price?.toFixed(2).padStart(6)} - $${etf.max_price?.toFixed(2).padStart(6)} | ` +
                        `${totalReturn.toFixed(1).padStart(6)}%`
                    );
                }

                // Show recent prices (last 5 trading days)
                console.log('\n📈 RECENT PRICES (Last 5 Trading Days):');
                db.all(`
                    SELECT DISTINCT date 
                    FROM prices 
                    ORDER BY date DESC 
                    LIMIT 5
                `, (err, recentDates) => {
                    if (err) {
                        console.error('❌ Error getting recent dates:', err);
                        return;
                    }

                    const dates = recentDates.map(r => r.date);
                    
                    console.log('\nSymbol | ' + dates.map(d => d.substring(5).padStart(10)).join(' | '));
                    console.log('-'.repeat(10 + dates.length * 13));

                    db.all(`
                        SELECT 
                            symbol,
                            date,
                            close
                        FROM prices 
                        WHERE date IN (${dates.map(() => '?').join(',')})
                        ORDER BY symbol, date DESC
                    `, dates, (err, recentPrices) => {
                        if (err) {
                            console.error('❌ Error getting recent prices:', err);
                            return;
                        }

                        const pricesBySymbol = {};
                        for (const price of recentPrices) {
                            if (!pricesBySymbol[price.symbol]) {
                                pricesBySymbol[price.symbol] = {};
                            }
                            pricesBySymbol[price.symbol][price.date] = price.close;
                        }

                        for (const [symbol, prices] of Object.entries(pricesBySymbol)) {
                            const priceStr = dates.map(date => {
                                const price = prices[date];
                                return price ? `$${price.toFixed(2)}`.padStart(10) : 'N/A'.padStart(10);
                            }).join(' | ');
                            console.log(`${symbol.padEnd(6)} | ${priceStr}`);
                        }

                        // Show yearly performance summary
                        console.log('\n📅 YEARLY PERFORMANCE SUMMARY:');
                        db.all(`
                            SELECT 
                                symbol,
                                strftime('%Y', date) as year,
                                COUNT(*) as trading_days,
                                MIN(close) as year_low,
                                MAX(close) as year_high,
                                (SELECT close FROM prices p2 WHERE p2.symbol = prices.symbol AND strftime('%Y', p2.date) = strftime('%Y', prices.date) ORDER BY date ASC LIMIT 1) as year_start,
                                (SELECT close FROM prices p3 WHERE p3.symbol = prices.symbol AND strftime('%Y', p3.date) = strftime('%Y', prices.date) ORDER BY date DESC LIMIT 1) as year_end
                            FROM prices 
                            GROUP BY symbol, strftime('%Y', date)
                            ORDER BY symbol, year
                        `, (err, yearlyData) => {
                            if (err) {
                                console.error('❌ Error getting yearly data:', err);
                                return;
                            }

                            let currentSymbol = '';
                            for (const data of yearlyData) {
                                if (data.symbol !== currentSymbol) {
                                    currentSymbol = data.symbol;
                                    console.log(`\n🏷️  ${data.symbol}:`);
                                    console.log('Year | Days | Year Low | Year High | Start    | End      | Return %');
                                    console.log('-'.repeat(65));
                                }

                                const yearReturn = data.year_start ? ((data.year_end - data.year_start) / data.year_start * 100) : 0;
                                console.log(
                                    `${data.year} | ${data.trading_days.toString().padStart(4)} | ` +
                                    `$${data.year_low.toFixed(2).padStart(7)} | ` +
                                    `$${data.year_high.toFixed(2).padStart(8)} | ` +
                                    `$${data.year_start.toFixed(2).padStart(7)} | ` +
                                    `$${data.year_end.toFixed(2).padStart(7)} | ` +
                                    `${yearReturn.toFixed(1).padStart(7)}%`
                                );
                            }

                            console.log('\n' + '='.repeat(80));
                            console.log('✅ DATA SUMMARY COMPLETE');
                            console.log(`📊 Total: ${stats.total_records} records across ${stats.etf_count} ETFs over ${endYear - startYear + 1} years`);
                            console.log('='.repeat(80));
                            
                            db.close();
                            resolve();
                        });
                    });
                });
            });
        });
    });
}

// Run the summary
showDataSummary().catch(console.error);