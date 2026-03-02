# GraphQL Anomaly Detection Middleware

> ML-powered security middleware that detects and blocks malicious GraphQL queries in real time.

## Architecture

```
Client → JS Middleware (Express/Apollo)
              ↓  feature vector (HTTP POST)
         Python ML Service (FastAPI)
              ↓  anomaly report JSON
         JS Middleware  →  Block (400) or Pass to GraphQL
```

**Two services — best of both worlds:**
| Service | Stack | Responsibility |
|---|---|---|
| `ml-service/` | Python · FastAPI · scikit-learn | Train + serve ML models |
| `middleware/`  | Node.js · Express · Apollo 4   | Parse queries, call ML, enforce decisions |

---

## Detection Coverage

| Threat | Method |
|---|---|
| Abnormal query depth | **Rule** — `max_depth > 7` |
| Alias abuse | **Rule** — `alias_count > 10` |
| Introspection abuse | **Rule** — `__schema`/`__type` detection |
| Excessive nesting | **Isolation Forest** (structural feature) |
| Field explosion | **Isolation Forest** (field count feature) |
| High resolver cost | **Isolation Forest** (cost feature) |
| Data exfiltration | **Isolation Forest** (field entropy + sensitive fields) |
| Query frequency | **EWMA** per-IP sliding window |

Scores are combined as a weighted ensemble: `55% structural + 25% frequency + 20% rules`.

---

## Quick Start

### 1 — Python ML Service

```bash
cd ml-service
pip install -r requirements.txt

# First run trains models automatically (~5 s)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Service available at `http://localhost:8000`  
Swagger docs at `http://localhost:8000/docs`

### 2 — JS Middleware + GraphQL Server

```bash
cd middleware
npm install
npm start
```

- GraphQL endpoint: `http://localhost:4000/graphql`
- Real-time dashboard: `http://localhost:4000/dashboard`
- Health check: `http://localhost:4000/health`

### Configuration (middleware/.env)

| Variable | Default | Description |
|---|---|---|
| `ML_SERVICE_URL` | `http://localhost:8000` | Python service URL |
| `BLOCK_THRESHOLD` | `0.6` | Score above this → blocked |
| `ALERT_ONLY` | `false` | Log only, don't block (shadow mode) |
| `MAX_QUERY_DEPTH` | `7` | Max allowed nesting depth |
| `MAX_ALIAS_COUNT` | `10` | Max aliases per query |
| `BLOCK_INTROSPECTION` | `false` | Block all introspection queries |

---

## Testing

```bash
# Python (from ml-service/)
pytest tests/ -v

# JavaScript (from middleware/)
npm test
```

---

## Example curl Tests

```bash
# ✅ Normal query — should pass
curl -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ users { id name email } }"}'

# 🚨 Deep nesting — should be blocked
curl -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ a { b { c { d { e { f { g { h { } } } } } } } } }"}'

# 🚨 Alias abuse — should be blocked
curl -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ a1:users{id} a2:users{id} a3:users{id} a4:users{id} a5:users{id} a6:users{id} a7:users{id} a8:users{id} a9:users{id} a10:users{id} a11:users{id} }"}'

# 🚨 Introspection — flagged (or blocked if BLOCK_INTROSPECTION=true)
curl -X POST http://localhost:4000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ __schema { types { name fields { name } } } }"}'
```

---

## API — Python ML Service

### `POST /analyze`
Accepts the feature vector, returns a full anomaly report.

**Request body:**
```json
{
  "max_depth": 3, "total_fields": 5, "unique_fields": 5,
  "alias_count": 0, "introspection_count": 0, "fragment_count": 0,
  "estimated_cost": 15, "payload_size": 80,
  "field_entropy": 1.5, "nesting_variance": 0.3,
  "client_ip": "127.0.0.1", "timestamp": 1709123456
}
```

**Response:**
```json
{
  "ensemble_score": 0.12,
  "is_anomaly": false,
  "threshold": 0.6,
  "component_scores": { "structural": 0.1, "frequency": 0.05, "rules": 0.0 },
  "rule_violations": {},
  "frequency_detail": { "current_rate_per_min": 2.0, "ewma_baseline": 1.8, "total_requests": 10 }
}
```

### `POST /train`
Re-trains all models from scratch (drops cached `.pkl` files).

### `GET /health`
Returns `{ "status": "ok", "uptime_seconds": N }`.

### `GET /metrics`
Returns aggregate stats (total analyzed, blocked, block rate).

---

## Project Structure

```
GraphQl/
├── ml-service/
│   ├── main.py                     # FastAPI app
│   ├── scorer.py                   # Weighted ensemble scorer
│   ├── requirements.txt
│   ├── detectors/
│   │   ├── structural_detector.py  # Isolation Forest
│   │   └── frequency_detector.py   # EWMA per-IP
│   ├── ml/
│   │   ├── training_data.py        # Synthetic data generator
│   │   └── trainer.py              # Train + persist models
│   ├── models/                     # Auto-created .pkl files
│   └── tests/
│       └── test_detectors.py
└── middleware/
    ├── src/
    │   ├── index.js                # Express server entry
    │   ├── logger.js               # Winston logger
    │   ├── schema.js               # Demo GraphQL schema
    │   └── middleware/
    │       ├── anomalyMiddleware.js # Core middleware
    │       ├── queryParser.js      # AST parser
    │       └── featureExtractor.js # Feature extraction
    ├── public/
    │   └── dashboard.html          # Real-time dashboard
    ├── tests/
    │   └── middleware.test.js
    ├── package.json
    └── .env
```
