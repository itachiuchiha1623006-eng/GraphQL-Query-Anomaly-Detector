// queryParser.js
// Parses a raw GraphQL query string into an AST using graphql-js.

const { parse, validate } = require('graphql');
const logger = require('../logger');

/**
 * Parse the raw query string to a DocumentNode AST.
 * Returns null if parsing fails (malformed GraphQL).
 */
function parseQuery(queryString) {
    if (!queryString || typeof queryString !== 'string') return null;
    try {
        return parse(queryString);
    } catch (err) {
        logger.warn('GraphQL parse error', { error: err.message });
        return null;
    }
}

module.exports = { parseQuery };
