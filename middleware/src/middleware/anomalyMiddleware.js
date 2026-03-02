// anomalyMiddleware.js
// Core Express middleware: parses the GraphQL query, extracts features,
// calls the Python ML service, and blocks or passes the request.

require('dotenv').config();
const axios = require('axios');
const { parseQuery } = require('./queryParser');
const { extractFeatures } = require('./featureExtractor');
const logger = require('../logger');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const BLOCK_THRESHOLD = parseFloat(process.env.BLOCK_THRESHOLD || '0.6');
const STRUCTURAL_THRESHOLD = parseFloat(process.env.STRUCTURAL_THRESHOLD || '0.72');
const ALERT_ONLY = process.env.ALERT_ONLY === 'true';


// SSE event emitter so the dashboard can stream anomaly events
const EventEmitter = require('events');
const anomalyEvents = new EventEmitter();
anomalyEvents.setMaxListeners(100);

/**
 * Main Express middleware factory.
 * Usage: app.use(anomalyMiddleware())
 */
function anomalyMiddleware() {
    return async function (req, res, next) {
        // Only inspect GraphQL endpoints (POST with JSON body containing a query)
        if (req.method !== 'POST' || !req.body?.query) return next();

        const rawQuery = req.body.query;
        const clientIp = req.headers['x-forwarded-for']?.split(',')[0].trim()
            || req.socket.remoteAddress
            || '0.0.0.0';

        // 1. Parse query into AST
        const ast = parseQuery(rawQuery);
        if (!ast) {
            // Malformed query — let Apollo handle the parse error
            return next();
        }

        // 2. Extract feature vector
        const features = extractFeatures(ast, rawQuery, clientIp);

        let report = null;
        try {
            // 3. Call Python ML service
            const response = await axios.post(`${ML_SERVICE_URL}/analyze`, features, {
                timeout: 3000, // 3 s hard limit — don't stall the request
            });
            report = response.data;
        } catch (err) {
            // ML service down → fail open (log + continue) to avoid breaking the API
            logger.error('ML service unreachable — failing open', {
                error: err.message,
                clientIp,
            });
            return next();
        }

        // 4. Determine block reason
        const hasRuleViolations = Object.keys(report.rule_violations || {}).length > 0;
        const exceedsMLThreshold = report.ensemble_score >= BLOCK_THRESHOLD;
        // Structural-only anomalies: IF score alone is high enough even without rules/frequency
        // (ensemble max without rules/freq = 0.55 × structural, so 0.72 structural → 0.396 ensemble, still below 0.6)
        // This separate check allows the Isolation Forest to block by itself.
        const highStructuralScore = (report.component_scores?.structural || 0) >= STRUCTURAL_THRESHOLD;

        const shouldBlock = hasRuleViolations || exceedsMLThreshold || highStructuralScore;
        const blockReason = hasRuleViolations ? 'RULE_VIOLATION'
            : highStructuralScore ? 'ML_STRUCTURAL'
                : 'ML_ANOMALY';

        // 5. Log the analysis result
        const logPayload = {
            clientIp,
            queryName: features.query_name,
            ensembleScore: report.ensemble_score,
            shouldBlock,
            blockReason: shouldBlock ? blockReason : null,
            ruleViolations: report.rule_violations,
            componentScores: report.component_scores,
        };

        if (shouldBlock) {
            logger.warn('🚨 Anomalous query detected', logPayload);
        } else {
            logger.info('✅ Query passed', { clientIp, queryName: features.query_name, score: report.ensemble_score });
        }

        // 6. Emit event to dashboard subscribers
        anomalyEvents.emit('analysis', {
            ...logPayload,
            isAnomaly: shouldBlock,
            timestamp: new Date().toISOString(),
            rawQuery: rawQuery.slice(0, 200),
        });

        // 7. Block or pass
        if (shouldBlock && !ALERT_ONLY) {
            return res.status(400).json({
                errors: [
                    {
                        message: hasRuleViolations
                            ? `Query blocked: rule violation detected.`
                            : 'Query blocked: ML anomaly score exceeded threshold.',
                        extensions: {
                            code: 'ANOMALY_DETECTED',
                            block_reason: blockReason,
                            ensemble_score: report.ensemble_score,
                            threshold: BLOCK_THRESHOLD,
                            component_scores: report.component_scores,
                            rule_violations: report.rule_violations,
                        },
                    },
                ],
            });
        }

        // Attach report to request so resolvers can access it if needed
        req.anomalyReport = report;
        return next();
    };
}

module.exports = { anomalyMiddleware, anomalyEvents };
