// tests/middleware.test.js
// End-to-end Jest + supertest tests for the GraphQL anomaly middleware.
// Starts a minimal Express server with a mock ML service.

const request = require('supertest');
const express = require('express');
const bodyParser = require('body-parser');

// ── Mock the ML service axios call ─────────────────────────────────────────
jest.mock('axios', () => ({
    post: jest.fn(),
    get: jest.fn(),
}));
const axios = require('axios');

// ── Import middleware modules ──────────────────────────────────────────────
const { anomalyMiddleware } = require('../src/middleware/anomalyMiddleware');
const { extractFeatures } = require('../src/middleware/featureExtractor');
const { parseQuery } = require('../src/middleware/queryParser');

// ── Build a minimal test app ───────────────────────────────────────────────
function buildApp(mlResponse) {
    axios.post.mockResolvedValue({ data: mlResponse });
    const app = express();
    app.use(bodyParser.json());
    app.use('/graphql', anomalyMiddleware());
    // Dummy endpoint — if middleware passes, returns 200
    app.post('/graphql', (req, res) => res.json({ data: { users: [] } }));
    return app;
}

const PASS_RESPONSE = {
    ensemble_score: 0.1,
    is_anomaly: false,
    threshold: 0.6,
    component_scores: { structural: 0.1, frequency: 0.05, rules: 0.0 },
    rule_violations: {},
    frequency_detail: { current_rate_per_min: 2, ewma_baseline: 2, total_requests: 5 },
    features_received: {},
};

const BLOCK_RESPONSE = {
    ensemble_score: 0.85,
    is_anomaly: true,
    threshold: 0.6,
    component_scores: { structural: 0.9, frequency: 0.8, rules: 0.7 },
    rule_violations: { depth: 'max_depth=20 > 7' },
    frequency_detail: { current_rate_per_min: 120, ewma_baseline: 5, total_requests: 200 },
    features_received: {},
};

// ── Tests ──────────────────────────────────────────────────────────────────

describe('Anomaly Middleware — E2E', () => {
    beforeEach(() => { process.env.ALERT_ONLY = 'false'; });

    test('passes a normal query through', async () => {
        const app = buildApp(PASS_RESPONSE);
        const res = await request(app)
            .post('/graphql')
            .send({ query: '{ users { id name } }' });
        expect(res.status).toBe(200);
        expect(res.body.data).toBeDefined();
    });

    test('blocks an anomalous query with 400', async () => {
        const app = buildApp(BLOCK_RESPONSE);
        // Valid deeply nested GraphQL query (each level has a real field)
        const deepQuery = '{ l1 { l2 { l3 { l4 { l5 { l6 { l7 { l8 { id } } } } } } } } }';
        const res = await request(app)
            .post('/graphql')
            .send({ query: deepQuery });
        expect(res.status).toBe(400);
        expect(res.body.errors[0].extensions.code).toBe('ANOMALY_DETECTED');
        expect(res.body.errors[0].extensions.ensemble_score).toBeGreaterThan(0.6);
    });

    test('includes rule_violations in block response', async () => {
        const app = buildApp(BLOCK_RESPONSE);
        const res = await request(app)
            .post('/graphql')
            .send({ query: '{ x }' });
        expect(res.body.errors[0].extensions.rule_violations).toHaveProperty('depth');
    });

    test('passes when ALERT_ONLY=true even for anomaly', async () => {
        process.env.ALERT_ONLY = 'true';
        // Need to re-require to pick up env change
        jest.resetModules();
        const { anomalyMiddleware: mw } = require('../src/middleware/anomalyMiddleware');
        axios.post.mockResolvedValue({ data: BLOCK_RESPONSE });
        const app = express();
        app.use(bodyParser.json());
        app.use('/graphql', mw());
        app.post('/graphql', (req, res) => res.json({ data: {} }));
        const res = await request(app).post('/graphql').send({ query: '{ x }' });
        expect(res.status).toBe(200);
    });

    test('fails open when ML service is unreachable', async () => {
        axios.post.mockRejectedValue(new Error('ECONNREFUSED'));
        const app = express();
        app.use(bodyParser.json());
        app.use('/graphql', anomalyMiddleware());
        app.post('/graphql', (req, res) => res.json({ data: {} }));
        const res = await request(app).post('/graphql').send({ query: '{ users { id } }' });
        expect(res.status).toBe(200); // fail open
    });

    test('skips non-GraphQL requests', async () => {
        const app = buildApp(PASS_RESPONSE);
        const res = await request(app).get('/graphql');
        // GET doesn't match middleware intercept, goes straight to next (404 from our minimal app)
        expect(res.status).not.toBe(400);
    });
});

// ── Feature Extractor Unit Tests ───────────────────────────────────────────

describe('Feature Extractor', () => {
    test('extracts correct max_depth for simple query', () => {
        const ast = parseQuery('{ users { id name } }');
        const f = extractFeatures(ast, '{ users { id name } }');
        expect(f.max_depth).toBe(2);
        expect(f.total_fields).toBe(3); // users, id, name
        expect(f.alias_count).toBe(0);
    });

    test('detects aliases', () => {
        const q = '{ a1: users { id } a2: users { id } }';
        const ast = parseQuery(q);
        const f = extractFeatures(ast, q);
        expect(f.alias_count).toBe(2);
    });

    test('counts introspection fields', () => {
        const q = '{ __schema { types { name } } }';
        const ast = parseQuery(q);
        const f = extractFeatures(ast, q);
        expect(f.introspection_count).toBeGreaterThan(0);
    });

    test('payload_size matches byte length', () => {
        const q = '{ users { id } }';
        const ast = parseQuery(q);
        const f = extractFeatures(ast, q);
        expect(f.payload_size).toBe(Buffer.byteLength(q, 'utf8'));
    });

    test('field_entropy > 0 for diverse field set', () => {
        const q = '{ users { id email name role posts { title body } } }';
        const ast = parseQuery(q);
        const f = extractFeatures(ast, q);
        expect(f.field_entropy).toBeGreaterThan(0);
    });

    test('handles malformed query gracefully', () => {
        const ast = parseQuery('not valid graphql !!!');
        expect(ast).toBeNull();
    });
});
