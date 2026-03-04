// src/index.js – Express + Apollo Server 4 with anomaly middleware

require('dotenv').config();
const express = require('express');
const path = require('path');
const { ApolloServer } = require('@apollo/server');
const { expressMiddleware } = require('@apollo/server/express4');
const { typeDefs, resolvers } = require('./schema');
const { anomalyMiddleware, anomalyEvents } = require('./middleware/anomalyMiddleware');
const logger = require('./logger');

const PORT = parseInt(process.env.PORT || '4000', 10);

async function start() {
    const app = express();

    // ── Static files ────────────────────────────────────────────────────────────
    app.use(express.static(path.join(__dirname, '..', 'public')));

    // ── Dashboard ───────────────────────────────────────────────────────────────
    app.get('/dashboard', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'public', 'dashboard.html'));
    });

    // ── E-Commerce Store ────────────────────────────────────────────────────────
    app.get('/shop', (req, res) => res.redirect('/shop/'));
    app.get('/shop/', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'public', 'shop', 'index.html'));
    });

    // ── Attack Demo ─────────────────────────────────────────────────────────────
    app.get('/attack-demo', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'public', 'attack-demo.html'));
    });

    // ── Query Analyzer Lab ───────────────────────────────────────────────────────
    app.get('/query-lab', (req, res) => {
        res.sendFile(path.join(__dirname, '..', 'public', 'query-lab.html'));
    });

    // ── Server-Sent Events ──────────────────────────────────────────────────────
    app.get('/events', (req, res) => {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no'); // disable nginx buffering if behind proxy
        res.flushHeaders();

        // Send a heartbeat every 15s to keep connection alive
        const heartbeat = setInterval(() => res.write(': heartbeat\n\n'), 15000);

        const send = (data) => {
            res.write(`data: ${JSON.stringify(data)}\n\n`);
        };

        anomalyEvents.on('analysis', send);

        req.on('close', () => {
            clearInterval(heartbeat);
            anomalyEvents.off('analysis', send);
        });
    });

    // ── Health ──────────────────────────────────────────────────────────────────
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

    // ── Apollo Server ───────────────────────────────────────────────────────────
    const apollo = new ApolloServer({ typeDefs, resolvers });
    await apollo.start();

    // ── /graphql route: anomaly middleware → express.json() → Apollo ────────────
    // IMPORTANT: express.json() must be applied at the route level for Apollo 4,
    // and our anomaly middleware must come BEFORE express.json() so it can read
    // the raw body stream before it's consumed.
    app.post(
        '/graphql',
        // Step 1: Buffer the raw body into req.rawBody so anomalyMiddleware can read it
        (req, res, next) => {
            let data = '';
            req.setEncoding('utf8');
            req.on('data', (chunk) => { data += chunk; });
            req.on('end', () => {
                req.rawBody = data;
                try {
                    req.body = JSON.parse(data);
                } catch {
                    req.body = {};
                }
                next();
            });
        },
        // Step 2: Anomaly detection (req.body.query is now available)
        anomalyMiddleware(),
        // Step 3: Apollo (expressMiddleware handles the rest; skip its body parsing
        //         by providing the already-parsed body via req.body)
        expressMiddleware(apollo, {
            context: async ({ req }) => ({
                anomalyReport: req.anomalyReport || null,
            }),
        })
    );

    // Also handle GET /graphql (GraphQL Playground / introspection via GET)
    app.get('/graphql', expressMiddleware(apollo));

    app.listen(PORT, () => {
        logger.info(`🚀 GraphQL server ready at http://localhost:${PORT}/graphql`);
        logger.info(`📊 Anomaly dashboard at  http://localhost:${PORT}/dashboard`);
    });
}

start().catch((err) => {
    logger.error('Failed to start server', { error: err.message });
    process.exit(1);
});
