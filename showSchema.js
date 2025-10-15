const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('./etf_data.db');

console.log('=== ETF-app-1w DATABASE STRUCTURE ===\n');

db.serialize(() => {
    // Get all table names
    db.all("SELECT name FROM sqlite_master WHERE type='table'", (err, tables) => {
        if (err) {
            console.error('Error getting tables:', err);
            return;
        }
        
        console.log('📋 TABLES FOUND:', tables.map(t => t.name).join(', '));
        console.log('\n');
        
        let completed = 0;
        
        tables.forEach(table => {
            console.log(`📊 TABLE: ${table.name.toUpperCase()}`);
            console.log('-'.repeat(70));
            
            // Get table schema
            db.all(`PRAGMA table_info(${table.name})`, (err, columns) => {
                if (err) {
                    console.error(`Error getting schema for ${table.name}:`, err);
                } else {
                    console.log('COLUMN NAME      TYPE          CONSTRAINTS');
                    console.log('-'.repeat(70));
                    
                    columns.forEach(col => {
                        let constraints = [];
                        if (col.pk) constraints.push('PRIMARY KEY');
                        if (col.notnull) constraints.push('NOT NULL');
                        if (col.dflt_value) constraints.push(`DEFAULT ${col.dflt_value}`);
                        
                        const name = col.name.padEnd(16);
                        const type = col.type.padEnd(13);
                        const cons = constraints.join(', ');
                        
                        console.log(`${name} ${type} ${cons}`);
                    });
                }
                
                // Get row count
                db.get(`SELECT COUNT(*) as count FROM ${table.name}`, (err, result) => {
                    if (!err && result) {
                        console.log(`\n📈 TOTAL RECORDS: ${result.count.toLocaleString()}`);
                    }
                    
                    // Show sample data
                    db.all(`SELECT * FROM ${table.name} LIMIT 3`, (err, samples) => {
                        if (!err && samples && samples.length > 0) {
                            console.log('\n📋 SAMPLE DATA:');
                            console.log(JSON.stringify(samples, null, 2));
                        }
                        
                        console.log('\n' + '='.repeat(70) + '\n');
                        
                        completed++;
                        if (completed === tables.length) {
                            db.close();
                            console.log('Database schema analysis complete.');
                        }
                    });
                });
            });
        });
    });
});