// src/index.js – Express + Apollo Server 4 entry point with anomaly middleware

require('dotenv').config();
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { typeDefs, resolvers } = require('./schema');
const { anomalyMiddleware, anomalyEvents } = require('./middleware/anomalyMiddleware');
const logger = require('./logger');

const PORT = parseInt(process.env.PORT || '4000', 10);

async function start() {
    const app = express();

    // ── Body parsing ──────────────────────────────────────────────────────────
    app.use(bodyParser.json({ limit: '1mb' }));

    // ── Static files (dashboard) ──────────────────────────────────────────────
    app.use(express.static(path.join(__dirname, '..', 'public')));

    // ── Dashboard route ───────────────────────────────────────────────────────
    app.get('/dashboard', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'public', 'dashboard.html'));
    });

    // ── Server-Sent Events stream for dashboard ───────────────────────────────
    app.get('/events', (req, res) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        const send = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
        anomalyEvents.on('analysis', send);

        req.on('close', () => {
            anomalyEvents.off('analysis', send);
        });
    });

    // ── Anomaly detection middleware (runs BEFORE Apollo) ─────────────────────
    app.use('/graphql', anomalyMiddleware());

    // ── Apollo Server ─────────────────────────────────────────────────────────
    const apollo = new ApolloServer({ typeDefs, resolvers });
    await apollo.start();

    app.use(
        '/graphql',
        expressMiddleware(apollo, {
            context: async ({ req }) => ({
                anomalyReport: req.anomalyReport || null,
            }),
        })
    );

    // ── Health + metrics proxy ────────────────────────────────────────────────
    app.get('/health', async (req, res) => {
        try {
            const axios = require('axios');
            const ml = await axios.get(
                `${process.env.ML_SERVICE_URL || 'http://localhost:8000'}/health`,
                { timeout: 2000 }
            );
            res.json({ middleware: 'ok', ml_service: ml.data });
        } catch {
            res.json({ middleware: 'ok', ml_service: 'unreachable' });
        }
    });

    app.listen(PORT, () => {
        logger.info(`🚀 GraphQL server ready at http://localhost:${PORT}/graphql`);
        logger.info(`📊 Anomaly dashboard at  http://localhost:${PORT}/dashboard`);
    });
}

start().catch((err) => {
    logger.error('Failed to start server', { error: err.message });
    process.exit(1);
});
