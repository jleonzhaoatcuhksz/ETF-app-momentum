const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('etf_data.db');

console.log('🔍 Checking database structure...');

// Check tables
db.all("SELECT name FROM sqlite_master WHERE type='table'", (err, tables) => {
    if (err) {
        console.error('❌ Database error:', err);
        return;
    }
    
    console.log('📊 Available tables:', tables);
    
    if (tables.length === 0) {
        console.log('⚠️ No tables found in database');
        db.close();
        return;
    }
    
    // Check first few records from each table
    let completed = 0;
    tables.forEach(table => {
        db.all(`SELECT * FROM ${table.name} LIMIT 5`, (err, rows) => {
            if (err) {
                console.error(`❌ Error querying ${table.name}:`, err);
            } else {
                console.log(`✅ Table ${table.name}: ${rows.length} sample records`);
                if (rows.length > 0) {
                    console.log('   Sample columns:', Object.keys(rows[0]));
                }
            }
            
            completed++;
            if (completed === tables.length) {
                db.close();
                console.log('🎯 Database check complete');
            }
        });
    });
});