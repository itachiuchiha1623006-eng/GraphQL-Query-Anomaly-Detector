const fs = require('fs');
const path = require('path');

const IBM_CORPUS_PATH = path.join(__dirname, '../../ml-service/data/generated_corpus.json');
const OUTPUT_PATH = path.join(__dirname, '../../ml-service/data/adversarial_corpus.json');

console.log("Loading IBM normal corpus...");
const plainQueries = JSON.parse(fs.readFileSync(IBM_CORPUS_PATH, 'utf8'));

const ADVERSARIAL_COUNT = 5000;
const attackQueries = [];

function getRandomQuery() {
    return plainQueries[Math.floor(Math.random() * plainQueries.length)];
}

// 1. Deep Nesting: Inject extreme nesting deep into the query tree
function mutateDeepNesting(queryStr) {
    const nesting = "{ __schema { types { fields { type { fields { type { name } } } } } } }";
    return queryStr.replace('{', '{ ' + nesting);
}

// 2. Field Explosion: Enormous horizontal field selection
function mutateFieldExplosion(queryStr) {
    let payload = "";
    for(let i=0; i<80; i++) payload += ` field_${i} `;
    return queryStr.replace('{', '{ ' + payload);
}

// 3. Alias Abuse: Querying the same fields to bypass limits
function mutateAliasAbuse(queryStr) {
    let payload = "";
    for(let i=0; i<30; i++) payload += ` a${i}: id `;
    return queryStr.replace('{', '{ ' + payload);
}

// 4. Introspection Injection: Reconnaissance hiding inside a normal query
function mutateIntrospectionInjection(queryStr) {
    const nesting = "{ __schema { queryType { name fields { name type { name } } } } }";
    return queryStr.replace('{', '{ ' + nesting);
}

// 5. Fragment Explosion: Many fragments declared
function mutateFragmentExplosion(queryStr) {
    let payload = "";
    for(let i=0; i<10; i++) payload += ` ...F${i} `;
    let frags = "";
    for(let i=0; i<10; i++) frags += ` fragment F${i} on Query { id } `;
    return queryStr.replace('{', '{ ' + payload) + " " + frags;
}

// 6. Batch / Directive Amplification
function mutateBatchAmplification(queryStr) {
    // Just inject a ton of fake variables / inline fragments
    let payload = "";
    for(let i=0; i<15; i++) payload += ` ... on User { id name } `;
    return queryStr.replace('{', '{ ' + payload);
}

const strategies = [
    mutateDeepNesting, 
    mutateFieldExplosion, 
    mutateAliasAbuse, 
    mutateIntrospectionInjection, 
    mutateFragmentExplosion,
    mutateBatchAmplification
];

console.log("Generating adversarial attacks via IBM corpus mutation...");
for (let i = 0; i < ADVERSARIAL_COUNT; i++) {
    const strategy = strategies[Math.floor(Math.random() * strategies.length)];
    const base = getRandomQuery();
    attackQueries.push(strategy(base));
}

fs.writeFileSync(OUTPUT_PATH, JSON.stringify(attackQueries, null, 2));
console.log(`Saved ${attackQueries.length} adversarial queries to ${OUTPUT_PATH}`);
