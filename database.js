const sqlite3 = require('sqlite3').verbose();
const config = require('./config');
const logger = require('./logger');

class ETFDatabase {
    constructor() {
        this.db = null;
        this.isInitialized = false;
    }

    async initialize() {
        return new Promise((resolve, reject) => {
            try {
                logger.info('Initializing ETF data database...');
                
                this.db = new sqlite3.Database(config.database.path, (err) => {
                    if (err) {
                        logger.error('Failed to connect to database:', err);
                        reject(err);
                        return;
                    }
                    
                    // Enable WAL mode and optimize settings
                    this.db.run("PRAGMA journal_mode = WAL");
                    this.db.run("PRAGMA synchronous = NORMAL");
                    this.db.run("PRAGMA cache_size = 10000");
                    this.db.run("PRAGMA temp_store = memory");
                    
                    this.createTables()
                        .then(() => {
                            this.isInitialized = true;
                            logger.success('Database initialized successfully');
                            resolve(true);
                        })
                        .catch(reject);
                });
            } catch (error) {
                logger.error('Failed to initialize database:', error);
                reject(error);
            }
        });
    }

    async createTables() {
        return new Promise((resolve, reject) => {
            // ETF metadata table
            const createETFsTable = `
                CREATE TABLE IF NOT EXISTS etfs (
                    symbol TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    category TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            `;

            // Price data table optimized for time series data
            const createPricesTable = `
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close REAL NOT NULL,
                    sma_5d REAL,
                    monthly_trend REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date),
                    FOREIGN KEY (symbol) REFERENCES etfs (symbol)
                )
            `;

            // Performance indexes for fast queries
            const createIndexes = [
                'CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol)',
                'CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)',
                'CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)',
                'CREATE INDEX IF NOT EXISTS idx_prices_close ON prices(close)',
                'CREATE INDEX IF NOT EXISTS idx_etfs_category ON etfs(category)',
                'CREATE INDEX IF NOT EXISTS idx_etfs_sector ON etfs(sector)'
            ];

            let completed = 0;
            const totalOperations = 2 + createIndexes.length;
            
            const checkComplete = (err) => {
                if (err) {
                    reject(err);
                    return;
                }
                completed++;
                if (completed === totalOperations) {
                    logger.info('Database tables and indexes created successfully');
                    resolve();
                }
            };

            this.db.serialize(() => {
                this.db.run(createETFsTable, checkComplete);
                this.db.run(createPricesTable, checkComplete);
                
                // Add sma_5d column if it doesn't exist (for existing databases)
                this.db.run(`
                    ALTER TABLE prices ADD COLUMN sma_5d REAL
                `, (err) => {
                    // Ignore error if column already exists
                    if (err && !err.message.includes('duplicate column name')) {
                        logger.error('Error adding sma_5d column:', err);
                    } else if (!err) {
                        logger.info('Added sma_5d column to prices table');
                    }
                });
                
                // Add monthly_trend column if it doesn't exist (for existing databases)
                this.db.run(`
                    ALTER TABLE prices ADD COLUMN monthly_trend REAL
                `, (err) => {
                    // Ignore error if column already exists
                    if (err && !err.message.includes('duplicate column name')) {
                        logger.error('Error adding monthly_trend column:', err);
                    } else if (!err) {
                        logger.info('Added monthly_trend column to prices table');
                    }
                });
                
                createIndexes.forEach(indexSQL => {
                    this.db.run(indexSQL, checkComplete);
                });
            });
        });
    }

    async insertETF(etfData) {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare(`
                INSERT OR REPLACE INTO etfs (symbol, name, sector, category, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            `);
            
            stmt.run(
                etfData.symbol,
                etfData.name,
                etfData.sector,
                etfData.category,
                (err) => {
                    if (err) reject(err);
                    else resolve();
                }
            );
            stmt.finalize();
        });
    }

    async insertPricesBatch(pricesArray) {
        return new Promise((resolve, reject) => {
            if (!pricesArray || pricesArray.length === 0) {
                resolve();
                return;
            }

            this.db.serialize(() => {
                this.db.run("BEGIN TRANSACTION");
                
                const stmt = this.db.prepare(`
                    INSERT OR REPLACE INTO prices 
                    (symbol, date, close, sma_5d)
                    VALUES (?, ?, ?, ?)
                `);
                
                let completed = 0;
                const total = pricesArray.length;
                let hasError = false;
                
                for (const price of pricesArray) {
                    stmt.run(
                        price.symbol,
                        price.date,
                        price.close,
                        price.sma_5d || null,
                        (err) => {
                            if (err && !hasError) {
                                hasError = true;
                                this.db.run("ROLLBACK");
                                reject(err);
                                return;
                            }
                            
                            completed++;
                            if (completed === total && !hasError) {
                                this.db.run("COMMIT", (err) => {
                                    if (err) reject(err);
                                    else resolve();
                                });
                            }
                        }
                    );
                }
                
                stmt.finalize();
            });
        });
    }

    async getPriceData(symbol, startDate = null, endDate = null, limit = null) {
        return new Promise((resolve, reject) => {
            let query = 'SELECT * FROM prices WHERE symbol = ?';
            const params = [symbol];
            
            if (startDate) {
                query += ' AND date >= ?';
                params.push(startDate);
            }
            
            if (endDate) {
                query += ' AND date <= ?';
                params.push(endDate);
            }
            
            query += ' ORDER BY date ASC';
            
            if (limit) {
                query += ' LIMIT ?';
                params.push(limit);
            }
            
            this.db.all(query, params, (err, rows) => {
                if (err) reject(err);
                else resolve(rows || []);
            });
        });
    }

    async getETFList() {
        return new Promise((resolve, reject) => {
            this.db.all('SELECT * FROM etfs ORDER BY symbol', (err, rows) => {
                if (err) reject(err);
                else resolve(rows || []);
            });
        });
    }

    async getStats() {
        return new Promise((resolve, reject) => {
            this.db.get('SELECT COUNT(*) as count FROM etfs', (err, etfCount) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                this.db.get('SELECT COUNT(*) as count FROM prices', (err, priceCount) => {
                    if (err) {
                        reject(err);
                        return;
                    }
                    
                    this.db.get('SELECT MIN(date) as earliest, MAX(date) as latest FROM prices', (err, dateRange) => {
                        if (err) {
                            reject(err);
                            return;
                        }
                        
                        resolve({
                            totalETFs: etfCount.count,
                            totalPrices: priceCount.count,
                            dateRange: dateRange
                        });
                    });
                });
            });
        });
    }

    async getExistingDataRange(symbol) {
        return new Promise((resolve, reject) => {
            this.db.get(
                'SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(*) as count FROM prices WHERE symbol = ?',
                [symbol],
                (err, row) => {
                    if (err) reject(err);
                    else resolve(row || { earliest: null, latest: null, count: 0 });
                }
            );
        });
    }

    async calculate5DaySMA(symbol = null) {
        return new Promise((resolve, reject) => {
            let query = `
                SELECT symbol, date, close 
                FROM prices 
                ${symbol ? 'WHERE symbol = ?' : ''}
                ORDER BY symbol, date ASC
            `;
            
            const params = symbol ? [symbol] : [];
            
            this.db.all(query, params, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                // Group by symbol
                const dataBySymbol = {};
                rows.forEach(row => {
                    if (!dataBySymbol[row.symbol]) {
                        dataBySymbol[row.symbol] = [];
                    }
                    dataBySymbol[row.symbol].push(row);
                });
                
                // Calculate 5-day SMA for each symbol
                const updates = [];
                
                Object.entries(dataBySymbol).forEach(([sym, prices]) => {
                    for (let i = 4; i < prices.length; i++) { // Start from 5th day (index 4)
                        const last5Prices = prices.slice(i - 4, i + 1);
                        const sma5d = last5Prices.reduce((sum, p) => sum + p.close, 0) / 5;
                        
                        updates.push({
                            symbol: sym,
                            date: prices[i].date,
                            sma_5d: Math.round(sma5d * 100) / 100 // Round to 2 decimal places
                        });
                    }
                });
                
                // Update database with SMA values
                if (updates.length === 0) {
                    logger.info('No SMA updates needed');
                    resolve(0);
                    return;
                }
                
                this.db.serialize(() => {
                    this.db.run("BEGIN TRANSACTION");
                    
                    const updateStmt = this.db.prepare(`
                        UPDATE prices 
                        SET sma_5d = ? 
                        WHERE symbol = ? AND date = ?
                    `);
                    
                    let processed = 0;
                    let hasError = false;
                    
                    updates.forEach(update => {
                        updateStmt.run(
                            update.sma_5d,
                            update.symbol,
                            update.date,
                            (err) => {
                                if (err && !hasError) {
                                    hasError = true;
                                    this.db.run("ROLLBACK");
                                    reject(err);
                                    return;
                                }
                                
                                processed++;
                                if (processed === updates.length && !hasError) {
                                    this.db.run("COMMIT", (err) => {
                                        if (err) reject(err);
                                        else {
                                            logger.success(`Updated ${updates.length} records with 5-day SMA`);
                                            resolve(updates.length);
                                        }
                                    });
                                }
                            }
                        );
                    });
                    
                    updateStmt.finalize();
                });
            });
        });
    }

    async calculateMonthlyTrend(symbol = null) {
        return new Promise((resolve, reject) => {
            // Get data with sma_5d values
            let query = `
                SELECT symbol, date, sma_5d 
                FROM prices 
                WHERE sma_5d IS NOT NULL
                ${symbol ? 'AND symbol = ?' : ''}
                ORDER BY symbol, date ASC
            `;
            
            const params = symbol ? [symbol] : [];
            
            this.db.all(query, params, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                // Group by symbol
                const dataBySymbol = {};
                rows.forEach(row => {
                    if (!dataBySymbol[row.symbol]) {
                        dataBySymbol[row.symbol] = [];
                    }
                    dataBySymbol[row.symbol].push(row);
                });
                
                // Calculate monthly trend for each symbol (using ~22 trading days as monthly period)
                const updates = [];
                const monthlyPeriod = 22; // Approximate trading days in a month
                
                Object.entries(dataBySymbol).forEach(([sym, prices]) => {
                    for (let i = monthlyPeriod - 1; i < prices.length; i++) {
                        const currentSMA = prices[i].sma_5d;
                        const monthAgoSMA = prices[i - monthlyPeriod + 1].sma_5d;
                        
                        if (currentSMA && monthAgoSMA) {
                            // Calculate monthly momentum as percentage change
                            const monthlyTrend = ((currentSMA - monthAgoSMA) / monthAgoSMA) * 100;
                            
                            updates.push({
                                symbol: sym,
                                date: prices[i].date,
                                monthly_trend: Math.round(monthlyTrend * 100) / 100 // Round to 2 decimal places
                            });
                        }
                    }
                });
                
                // Update database with monthly trend values
                if (updates.length === 0) {
                    logger.info('No monthly trend updates needed');
                    resolve(0);
                    return;
                }
                
                this.db.serialize(() => {
                    this.db.run("BEGIN TRANSACTION");
                    
                    const updateStmt = this.db.prepare(`
                        UPDATE prices 
                        SET monthly_trend = ? 
                        WHERE symbol = ? AND date = ?
                    `);
                    
                    let processed = 0;
                    let hasError = false;
                    
                    updates.forEach(update => {
                        updateStmt.run(
                            update.monthly_trend,
                            update.symbol,
                            update.date,
                            (err) => {
                                if (err && !hasError) {
                                    hasError = true;
                                    this.db.run("ROLLBACK");
                                    reject(err);
                                    return;
                                }
                                
                                processed++;
                                if (processed === updates.length && !hasError) {
                                    this.db.run("COMMIT", (err) => {
                                        if (err) reject(err);
                                        else {
                                            logger.success(`Updated ${updates.length} records with monthly trend`);
                                            resolve(updates.length);
                                        }
                                    });
                                }
                            }
                        );
                    });
                    
                    updateStmt.finalize();
                });
            });
        });
    }

    close() {
        if (this.db) {
            this.db.close((err) => {
                if (err) {
                    logger.error('Error closing database:', err);
                } else {
                    logger.info('Database connection closed');
                }
            });
            this.isInitialized = false;
        }
    }
}

module.exports = new ETFDatabase();