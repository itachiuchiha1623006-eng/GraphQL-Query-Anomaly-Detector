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
const FREQUENCY_THRESHOLD = parseFloat(process.env.FREQUENCY_THRESHOLD || '0.7');
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
        // Structural-only: Isolation Forest score alone is high enough to block
        const highStructuralScore = (report.component_scores?.structural || 0) >= STRUCTURAL_THRESHOLD;
        // Frequency-only: rapid burst score alone is high enough to block
        // (max frequency contribution to ensemble = 0.25, never reaches 0.6 threshold alone)
        const highFrequencyScore = (report.component_scores?.frequency || 0) >= FREQUENCY_THRESHOLD
            && !report.frequency_detail?.warmup;

        const shouldBlock = hasRuleViolations || exceedsMLThreshold || highStructuralScore || highFrequencyScore;
        const blockReason = hasRuleViolations ? 'RULE_VIOLATION'
            : highFrequencyScore ? 'FREQUENCY_ATTACK'
                : highStructuralScore ? 'ML_STRUCTURAL'
                    : 'ML_ANOMALY';

        // 5. Determine Severity Level for multi-level logging
        let severity = 'INFO';
        if (hasRuleViolations || report.ensemble_score >= 0.85) {
            severity = 'FATAL';
        } else if (shouldBlock || report.ensemble_score >= 0.40) {
            severity = 'WARN';
        }

        // 6. Log the analysis result using multi-level severities
        const logPayload = {
            clientIp,
            queryName: features.query_name,
            ensembleScore: report.ensemble_score,
            shouldBlock,
            blockReason: shouldBlock ? blockReason : null,
            ruleViolations: report.rule_violations,
            componentScores: report.component_scores,
            severity,
        };

        if (severity === 'FATAL') {
            logger.error('💥 FATAL: Critical attack intercepted', logPayload);
        } else if (severity === 'WARN') {
            logger.warn(`🚨 WARN: ${shouldBlock ? 'Anomalous query blocked' : 'Suspicious but unblocked query'}`, logPayload);
        } else {
            logger.info('✅ INFO: Normal query passed', { clientIp, queryName: features.query_name, score: report.ensemble_score });
        }

        // 7. Emit event to dashboard subscribers with severity
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
                            ? 'Query blocked: rule violation detected.'
                            : highFrequencyScore
                                ? 'Query blocked: request rate spike detected.'
                                : 'Query blocked: ML anomaly score exceeded threshold.',
                        extensions: {
                            code: 'ANOMALY_DETECTED',
                            block_reason: blockReason,
                            ensemble_score: report.ensemble_score,
                            threshold: BLOCK_THRESHOLD,
                            component_scores: report.component_scores,
                            rule_violations: report.rule_violations,
                            frequency_detail: report.frequency_detail,
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
