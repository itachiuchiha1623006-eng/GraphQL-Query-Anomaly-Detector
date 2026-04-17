const { typeDefs } = require('../src/schema.js');
const { buildSchema, print } = require('graphql');
const generateQuery = require('ibm-graphql-query-generator');
const fs = require('fs');
const path = require('path');

// 1. Build the schema exactly as the real backend works
const schema = buildSchema(typeDefs);

// 2. Generate Corpus
const queries = [];
const NUM_QUERIES = 2000;

console.log(`Generating ${NUM_QUERIES} GraphQL queries using ibm-graphql-query-generator...`);

for (let i = 0; i < NUM_QUERIES; i++) {
  try {
      // Configuration for generating realistic/varied queries
      // We vary some configurations to get diverse payloads.
      const config = {
          depthProbability: Math.random() * 0.5 + 0.1,    // 0.1 to 0.6
          breadthProbability: Math.random() * 0.5 + 0.1,  // 0.1 to 0.6
          maxDepth: Math.floor(Math.random() * 5) + 2,    // 2 to 6
          ignoreOptionalArguments: Math.random() > 0.5,
          providePlaceholders: true,
          argumentsToIgnore: [],
          argumentsToConsider: [],
          providerMap: {},
          considerInterfaces: true,
          considerUnions: true,
      };

      const ast = generateQuery.generateRandomQuery(schema, config);
      if (ast && ast.queryDocument) {
          const queryString = print(ast.queryDocument);
          queries.push(queryString);
      }
  } catch (e) {
      console.error('Error generating a query:', e);
  }
}

// 3. Output to ml-service
const outPath = path.join(__dirname, '../../ml-service/data/generated_corpus.json');
fs.writeFileSync(outPath, JSON.stringify(queries, null, 2));

console.log(`Successfully generated ${queries.length} queries and wrote to ${outPath}`);
