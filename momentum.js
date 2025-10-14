// ETF Momentum Analysis Engine
const logger = require('./logger');

class MomentumAnalyzer {
    constructor() {
        this.indicators = {
            RSI_PERIOD: 14,
            MACD_FAST: 12,
            MACD_SLOW: 26,
            MACD_SIGNAL: 9,
            MA_SHORT: 20,
            MA_LONG: 50,
            MOMENTUM_PERIOD: 10
        };
    }

    /**
     * Calculate RSI (Relative Strength Index)
     * @param {Array} prices - Array of price objects with close values
     * @param {number} period - RSI period (default 14)
     * @returns {Object} RSI data
     */
    calculateRSI(prices, period = this.indicators.RSI_PERIOD) {
        if (prices.length < period + 1) {
            return { value: null, signal: 'insufficient_data' };
        }

        const closes = prices.map(p => p.close);
        const gains = [];
        const losses = [];

        // Calculate price changes
        for (let i = 1; i < closes.length; i++) {
            const change = closes[i] - closes[i - 1];
            gains.push(change > 0 ? change : 0);
            losses.push(change < 0 ? Math.abs(change) : 0);
        }

        // Calculate average gains and losses
        let avgGain = gains.slice(0, period).reduce((sum, gain) => sum + gain, 0) / period;
        let avgLoss = losses.slice(0, period).reduce((sum, loss) => sum + loss, 0) / period;

        const rsiValues = [];

        // Calculate RSI for each period
        for (let i = period; i < gains.length; i++) {
            if (avgLoss === 0) {
                rsiValues.push(100);
            } else {
                const rs = avgGain / avgLoss;
                const rsi = 100 - (100 / (1 + rs));
                rsiValues.push(rsi);
            }

            // Update averages using Wilder's smoothing
            avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
            avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
        }

        const currentRSI = rsiValues[rsiValues.length - 1];
        let signal = 'neutral';
        
        if (currentRSI > 70) signal = 'overbought';
        else if (currentRSI < 30) signal = 'oversold';
        else if (currentRSI > 50) signal = 'bullish';
        else signal = 'bearish';

        return {
            value: Math.round(currentRSI * 100) / 100,
            signal,
            history: rsiValues.slice(-30) // Last 30 values
        };
    }

    /**
     * Calculate MACD (Moving Average Convergence Divergence)
     * @param {Array} prices - Array of price objects
     * @returns {Object} MACD data
     */
    calculateMACD(prices) {
        const { MACD_FAST, MACD_SLOW, MACD_SIGNAL } = this.indicators;
        
        if (prices.length < MACD_SLOW + MACD_SIGNAL) {
            return { macd: null, signal: null, histogram: null, trend: 'insufficient_data' };
        }

        const closes = prices.map(p => p.close);
        
        // Calculate EMAs
        const emaFast = this.calculateEMA(closes, MACD_FAST);
        const emaSlow = this.calculateEMA(closes, MACD_SLOW);
        
        // Calculate MACD line
        const macdLine = [];
        for (let i = 0; i < emaFast.length; i++) {
            if (emaSlow[i] !== null && emaFast[i] !== null) {
                macdLine.push(emaFast[i] - emaSlow[i]);
            }
        }
        
        // Calculate signal line (EMA of MACD)
        const signalLine = this.calculateEMA(macdLine, MACD_SIGNAL);
        
        // Calculate histogram
        const histogram = [];
        for (let i = 0; i < macdLine.length; i++) {
            if (signalLine[i] !== null) {
                histogram.push(macdLine[i] - signalLine[i]);
            }
        }

        const currentMACD = macdLine[macdLine.length - 1];
        const currentSignal = signalLine[signalLine.length - 1];
        const currentHistogram = histogram[histogram.length - 1];

        let trend = 'neutral';
        if (currentMACD > currentSignal && currentHistogram > 0) {
            trend = 'bullish';
        } else if (currentMACD < currentSignal && currentHistogram < 0) {
            trend = 'bearish';
        }

        return {
            macd: Math.round(currentMACD * 10000) / 10000,
            signal: Math.round(currentSignal * 10000) / 10000,
            histogram: Math.round(currentHistogram * 10000) / 10000,
            trend,
            history: {
                macd: macdLine.slice(-30),
                signal: signalLine.slice(-30),
                histogram: histogram.slice(-30)
            }
        };
    }

    /**
     * Calculate Exponential Moving Average
     * @param {Array} values - Array of values
     * @param {number} period - EMA period
     * @returns {Array} EMA values
     */
    calculateEMA(values, period) {
        if (values.length < period) return [];
        
        const ema = [];
        const multiplier = 2 / (period + 1);
        
        // First EMA is SMA
        let sum = 0;
        for (let i = 0; i < period; i++) {
            sum += values[i];
        }
        ema.push(sum / period);
        
        // Calculate subsequent EMAs
        for (let i = period; i < values.length; i++) {
            const currentEMA = (values[i] * multiplier) + (ema[ema.length - 1] * (1 - multiplier));
            ema.push(currentEMA);
        }
        
        return ema;
    }

    /**
     * Calculate Simple Moving Average
     * @param {Array} values - Array of values
     * @param {number} period - MA period
     * @returns {Array} MA values
     */
    calculateSMA(values, period) {
        if (values.length < period) return [];
        
        const sma = [];
        for (let i = period - 1; i < values.length; i++) {
            const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
        return sma;
    }

    /**
     * Calculate price momentum
     * @param {Array} prices - Array of price objects
     * @param {number} period - Momentum period
     * @returns {Object} Momentum data
     */
    calculateMomentum(prices, period = this.indicators.MOMENTUM_PERIOD) {
        if (prices.length < period + 1) {
            return { value: null, signal: 'insufficient_data' };
        }

        const closes = prices.map(p => p.close);
        const currentPrice = closes[closes.length - 1];
        const previousPrice = closes[closes.length - 1 - period];
        
        const momentum = ((currentPrice - previousPrice) / previousPrice) * 100;
        
        let signal = 'neutral';
        if (momentum > 5) signal = 'strong_bullish';
        else if (momentum > 2) signal = 'bullish';
        else if (momentum < -5) signal = 'strong_bearish';
        else if (momentum < -2) signal = 'bearish';

        return {
            value: Math.round(momentum * 100) / 100,
            signal,
            period
        };
    }

    /**
     * Calculate moving average signals
     * @param {Array} prices - Array of price objects
     * @returns {Object} Moving average analysis
     */
    calculateMovingAverages(prices) {
        const { MA_SHORT, MA_LONG } = this.indicators;
        
        if (prices.length < MA_LONG) {
            return { signal: 'insufficient_data' };
        }

        const closes = prices.map(p => p.close);
        const maShort = this.calculateSMA(closes, MA_SHORT);
        const maLong = this.calculateSMA(closes, MA_LONG);
        
        const currentPrice = closes[closes.length - 1];
        const currentMAShort = maShort[maShort.length - 1];
        const currentMALong = maLong[maLong.length - 1];
        
        let signal = 'neutral';
        let trend = 'sideways';
        
        if (currentMAShort > currentMALong) {
            trend = 'uptrend';
            if (currentPrice > currentMAShort) {
                signal = 'bullish';
            }
        } else if (currentMAShort < currentMALong) {
            trend = 'downtrend';
            if (currentPrice < currentMAShort) {
                signal = 'bearish';
            }
        }

        return {
            ma_short: Math.round(currentMAShort * 100) / 100,
            ma_long: Math.round(currentMALong * 100) / 100,
            current_price: currentPrice,
            signal,
            trend,
            above_ma_short: currentPrice > currentMAShort,
            above_ma_long: currentPrice > currentMALong
        };
    }

    /**
     * Calculate volume analysis
     * @param {Array} prices - Array of price objects with volume
     * @returns {Object} Volume analysis
     */
    calculateVolumeAnalysis(prices) {
        if (prices.length < 20) {
            return { signal: 'insufficient_data' };
        }

        const volumes = prices.map(p => p.volume || 0).filter(v => v > 0);
        if (volumes.length === 0) {
            return { signal: 'no_volume_data' };
        }

        const avgVolume = volumes.slice(-20).reduce((sum, vol) => sum + vol, 0) / 20;
        const currentVolume = volumes[volumes.length - 1];
        const volumeRatio = currentVolume / avgVolume;

        let signal = 'normal';
        if (volumeRatio > 2) signal = 'high_volume';
        else if (volumeRatio > 1.5) signal = 'above_average';
        else if (volumeRatio < 0.5) signal = 'low_volume';

        return {
            current_volume: currentVolume,
            avg_volume: Math.round(avgVolume),
            volume_ratio: Math.round(volumeRatio * 100) / 100,
            signal
        };
    }

    /**
     * Calculate comprehensive momentum score
     * @param {Array} prices - Array of price objects
     * @returns {Object} Complete momentum analysis
     */
    analyzeMomentum(prices) {
        try {
            const rsi = this.calculateRSI(prices);
            const macd = this.calculateMACD(prices);
            const momentum = this.calculateMomentum(prices);
            const movingAverages = this.calculateMovingAverages(prices);
            const volume = this.calculateVolumeAnalysis(prices);

            // Calculate overall momentum score (0-100)
            let score = 50; // Neutral starting point
            
            // RSI contribution (±15 points)
            if (rsi.value !== null) {
                if (rsi.signal === 'bullish') score += 10;
                else if (rsi.signal === 'strong_bullish') score += 15;
                else if (rsi.signal === 'bearish') score -= 10;
                else if (rsi.signal === 'oversold') score += 5; // Oversold can be opportunity
                else if (rsi.signal === 'overbought') score -= 15;
            }

            // MACD contribution (±15 points)
            if (macd.trend === 'bullish') score += 15;
            else if (macd.trend === 'bearish') score -= 15;

            // Price momentum contribution (±20 points)
            if (momentum.signal === 'strong_bullish') score += 20;
            else if (momentum.signal === 'bullish') score += 10;
            else if (momentum.signal === 'bearish') score -= 10;
            else if (momentum.signal === 'strong_bearish') score -= 20;

            // Moving average contribution (±10 points)
            if (movingAverages.signal === 'bullish') score += 10;
            else if (movingAverages.signal === 'bearish') score -= 10;

            // Volume contribution (±5 points)
            if (volume.signal === 'high_volume' && score > 50) score += 5;
            else if (volume.signal === 'low_volume' && score < 50) score -= 3;

            // Ensure score stays within 0-100 range
            score = Math.max(0, Math.min(100, score));

            // Determine overall rating
            let rating = 'Neutral';
            if (score >= 80) rating = 'Very Bullish';
            else if (score >= 65) rating = 'Bullish';
            else if (score >= 55) rating = 'Slightly Bullish';
            else if (score <= 20) rating = 'Very Bearish';
            else if (score <= 35) rating = 'Bearish';
            else if (score <= 45) rating = 'Slightly Bearish';

            return {
                score: Math.round(score),
                rating,
                indicators: {
                    rsi,
                    macd,
                    momentum,
                    moving_averages: movingAverages,
                    volume
                },
                analysis_date: new Date().toISOString(),
                data_points: prices.length
            };

        } catch (error) {
            logger.error('Error in momentum analysis:', error);
            return {
                score: null,
                rating: 'Error',
                error: error.message
            };
        }
    }

    /**
     * Calculate performance metrics
     * @param {Array} prices - Array of price objects
     * @returns {Object} Performance metrics
     */
    calculatePerformance(prices) {
        if (prices.length < 2) {
            return { error: 'insufficient_data' };
        }

        const closes = prices.map(p => p.close);
        const dates = prices.map(p => p.date);
        
        const currentPrice = closes[closes.length - 1];
        const startPrice = closes[0];
        
        // Calculate returns for different periods
        const performance = {
            current_price: currentPrice,
            total_return: ((currentPrice - startPrice) / startPrice) * 100
        };

        // Calculate period returns if enough data
        const periods = [
            { name: '1d', days: 1 },
            { name: '1w', days: 7 },
            { name: '1m', days: 30 },
            { name: '3m', days: 90 },
            { name: '6m', days: 180 },
            { name: '1y', days: 365 }
        ];

        for (const period of periods) {
            const targetIndex = Math.max(0, closes.length - period.days);
            if (targetIndex < closes.length - 1) {
                const periodStartPrice = closes[targetIndex];
                performance[period.name] = ((currentPrice - periodStartPrice) / periodStartPrice) * 100;
            }
        }

        return performance;
    }
}

module.exports = new MomentumAnalyzer();