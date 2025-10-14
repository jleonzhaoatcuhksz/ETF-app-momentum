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
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL,
                    volume INTEGER,
                    adj_close REAL,
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
                    (symbol, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                `);
                
                let completed = 0;
                const total = pricesArray.length;
                let hasError = false;
                
                for (const price of pricesArray) {
                    stmt.run(
                        price.symbol,
                        price.date,
                        price.open,
                        price.high,
                        price.low,
                        price.close,
                        price.volume,
                        price.adj_close || price.close,
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