const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m'
};

class Logger {
    constructor() {
        this.level = 'info';
    }

    formatMessage(level, message, data = null) {
        const timestamp = new Date().toISOString();
        const prefix = `[${timestamp}] `;
        
        let formattedMessage = `${prefix}${message}`;
        if (data) {
            formattedMessage += ` ${JSON.stringify(data)}`;
        }
        
        return formattedMessage;
    }

    info(message, data = null) {
        const formatted = this.formatMessage('INFO', message, data);
        console.log(`${colors.cyan}ℹ️  INFO: ${colors.reset}${message}${data ? ' ' + JSON.stringify(data) : ''}`);
    }

    success(message, data = null) {
        const formatted = this.formatMessage('SUCCESS', message, data);
        console.log(`${colors.green}✅ SUCCESS: ${colors.reset}${message}${data ? ' ' + JSON.stringify(data) : ''}`);
    }

    warning(message, data = null) {
        const formatted = this.formatMessage('WARNING', message, data);
        console.log(`${colors.yellow}⚠️  WARNING: ${colors.reset}${message}${data ? ' ' + JSON.stringify(data) : ''}`);
    }

    error(message, data = null) {
        const formatted = this.formatMessage('ERROR', message, data);
        console.log(`${colors.red}❌ ERROR: ${colors.reset}${message}${data ? ' ' + JSON.stringify(data) : ''}`);
    }

    progress(current, total, item = '') {
        const percentage = Math.round((current / total) * 100);
        const bar = '█'.repeat(Math.floor(percentage / 5)) + '░'.repeat(20 - Math.floor(percentage / 5));
        process.stdout.write(`\r${colors.blue}📊 Progress: [${bar}] ${percentage}% (${current}/${total}) ${item}${colors.reset}`);
        
        if (current === total) {
            console.log(); // New line when complete
        }
    }
}

module.exports = new Logger();