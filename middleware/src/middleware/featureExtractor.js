// featureExtractor.js
// Walks a GraphQL AST and produces a numeric feature vector for the ML service.

const { Kind } = require('graphql');

// Sensitive field names that may indicate data exfiltration intent
const SENSITIVE_FIELDS = new Set([
    'password', 'token', 'secret', 'apiKey', 'api_key', 'creditCard',
    'credit_card', 'ssn', 'socialSecurity', 'privateKey', 'private_key',
    'accessToken', 'refreshToken', 'authToken', 'hash', 'salt', 'wallet',
    'bankAccount', 'cvv', 'pin',
]);

const INTROSPECTION_FIELDS = new Set([
    '__schema', '__type', '__typename', '__enumValue',
    '__inputValue', '__field', '__directive',
]);

/**
 
 * Compute Shannon entropy of an array of strings.
 * Higher entropy = more diverse / unusual field set.
 
 */

function shannonEntropy(names) {
    if (!names.length) return 0;
    const freq = {};
    for (const n of names) freq[n] = (freq[n] || 0) + 1;
    const total = names.length;
    return -Object.values(freq).reduce((sum, count) => {
        const p = count / total;
        return sum + p * Math.log2(p);
    }, 0);
}

/**
 * Recursively walk selection sets, collecting depth, fields, aliases, etc.
 */
function walkSelectionSet(selectionSet, depth, state) {
    if (!selectionSet || !selectionSet.selections) return;

    state.depthHistogram[depth] = (state.depthHistogram[depth] || 0) + 1;
    state.maxDepth = Math.max(state.maxDepth, depth);

    for (const selection of selectionSet.selections) {
        if (selection.kind === Kind.FIELD) {
            const name = selection.name.value;
            state.allFieldNames.push(name);
            state.totalFields += 1;
            state.uniqueFields.add(name);

            if (selection.alias) state.aliasCount += 1;
            if (INTROSPECTION_FIELDS.has(name)) state.introspectionCount += 1;
            if (SENSITIVE_FIELDS.has(name)) state.sensitiveCount += 1;

            // Cost model: each field costs 1 * depth multiplier
            state.estimatedCost += depth;

            if (selection.selectionSet) {
                walkSelectionSet(selection.selectionSet, depth + 1, state);
            }

        } else if (selection.kind === Kind.INLINE_FRAGMENT) {
            state.fragmentCount += 1;
            walkSelectionSet(selection.selectionSet, depth + 1, state);

        } else if (selection.kind === Kind.FRAGMENT_SPREAD) {
            state.fragmentCount += 1;
        }
    }
}

/**
 * Compute nesting variance from the depth histogram.
 */
function nestingVariance(histogram) {
    const entries = Object.entries(histogram);
    if (entries.length < 2) return 0;
    const values = entries.map(([depth]) => Number(depth));
    const counts = entries.map(([, count]) => count);
    const total = counts.reduce((a, b) => a + b, 0);
    const mean = values.reduce((s, v, i) => s + v * counts[i], 0) / total;
    const variance = values.reduce((s, v, i) => s + counts[i] * (v - mean) ** 2, 0) / total;
    return variance;
}

/**
 * Main entry point.
 * @param {DocumentNode} ast   Parsed GraphQL AST
 * @param {string} rawQuery    Original query string
 * @param {string} clientIp    Request IP
 * @returns {Object}           Feature vector (matches Python FeatureVector schema)
 */
function extractFeatures(ast, rawQuery, clientIp = '0.0.0.0') {
    const state = {
        maxDepth: 0,
        totalFields: 0,
        uniqueFields: new Set(),
        aliasCount: 0,
        introspectionCount: 0,
        fragmentCount: 0,
        estimatedCost: 0,
        allFieldNames: [],
        sensitiveCount: 0,
        depthHistogram: {},
    };

    if (ast && ast.definitions) {
        for (const def of ast.definitions) {
            if (def.selectionSet) {
                walkSelectionSet(def.selectionSet, 1, state);
            }
        }
    }

    const entropy = shannonEntropy(state.allFieldNames);
    const variance = nestingVariance(state.depthHistogram);

    // Boost entropy if sensitive fields are detected
    const fieldEntropy = entropy + state.sensitiveCount * 0.5;

    return {
        max_depth: state.maxDepth,
        total_fields: state.totalFields,
        unique_fields: state.uniqueFields.size,
        alias_count: state.aliasCount,
        introspection_count: state.introspectionCount,
        fragment_count: state.fragmentCount,
        estimated_cost: state.estimatedCost,
        payload_size: Buffer.byteLength(rawQuery || '', 'utf8'),
        field_entropy: parseFloat(fieldEntropy.toFixed(4)),
        nesting_variance: parseFloat(variance.toFixed(4)),
        client_ip: clientIp,
        timestamp: Date.now() / 1000,
        query_name: (ast?.definitions?.[0]?.name?.value) || '',
    };
}

module.exports = { extractFeatures };
