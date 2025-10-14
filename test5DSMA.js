// Test script to verify 5-day SMA calculation
const database = require('./database');
const logger = require('./logger');

async function test5DSMA() {
    try {
        await database.initialize();
        
        // Get sample data with 5D-SMA
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
        
        console.log('\n📊 Sample 5-Day SMA Data:');
        console.log('Symbol | Date       | Close   | 5D-SMA');
        console.log('-------|------------|---------|-------');
        
        sampleData.forEach(row => {
            console.log(
                `${row.symbol.padEnd(6)} | ${row.date} | $${row.close.toFixed(2).padStart(6)} | $${row.sma_5d.toFixed(2).padStart(6)}`
            );
        });
        
        // Get count of records with 5D-SMA
        const count = await new Promise((resolve, reject) => {
            database.db.get(`
                SELECT COUNT(*) as count 
                FROM prices 
                WHERE sma_5d IS NOT NULL
            `, (err, row) => {
                if (err) reject(err);
                else resolve(row.count);
            });
        });
        
        console.log(`\n✅ Total records with 5D-SMA: ${count.toLocaleString()}`);
        
    } catch (error) {
        console.error('Error:', error);
    } finally {
        database.close();
    }
}

test5DSMA();