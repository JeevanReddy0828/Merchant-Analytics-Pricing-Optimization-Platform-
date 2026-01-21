# API Documentation

## Overview

The Pricing Optimization API provides real-time ML-powered pricing recommendations and churn risk assessments for merchants.

**Base URL**: `http://localhost:8000`

## Authentication

Currently no authentication required (development mode).

## Endpoints

### Health Check

```
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

---

### Get Pricing Recommendation

```
POST /api/v1/pricing/recommend
```

Get optimal pricing recommendation for a single merchant.

**Request Body**:
```json
{
  "merchant_id": "M123456",
  "business_category": "E-commerce",
  "region": "North America",
  "pricing_tier": "standard",
  "monthly_volume_usd": 50000,
  "avg_transaction_size": 75.50,
  "account_age_months": 24,
  "cross_border_ratio": 0.15,
  "dispute_rate": 0.002,
  "refund_rate": 0.03,
  "churn_risk_score": 0.12
}
```

**Response**:
```json
{
  "merchant_id": "M123456",
  "current_tier": "standard",
  "recommended_rate": 0.0265,
  "current_rate": 0.029,
  "rate_change": -0.0025,
  "confidence": 0.85,
  "factors": {
    "volume_impact": -0.002,
    "churn_risk_impact": -0.00036,
    "cross_border_impact": 0.0015
  }
}
```

---

### Batch Pricing Recommendations

```
POST /api/v1/pricing/batch
```

Get pricing recommendations for multiple merchants.

**Request Body**:
```json
{
  "merchants": [
    { "merchant_id": "M001", ... },
    { "merchant_id": "M002", ... }
  ]
}
```

---

### Assess Churn Risk

```
POST /api/v1/churn/assess
```

Assess churn risk for a merchant.

**Request Body**: Same as pricing endpoint

**Response**:
```json
{
  "merchant_id": "M123456",
  "churn_probability": 0.23,
  "risk_tier": "MEDIUM",
  "revenue_at_risk": 312.50,
  "recommended_action": "Feature adoption email series"
}
```

---

### Get Model Metrics

```
GET /api/v1/model/metrics
```

Get current model performance metrics.

**Response**:
```json
{
  "pricing_model": {
    "status": "trained",
    "features": ["monthly_volume_usd", "churn_risk_score", ...]
  },
  "churn_model": {
    "status": "trained",
    "features": ["days_since_last_txn", "txn_frequency", ...]
  }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 503 | Model not trained |

## Running the API

```bash
# Install dependencies
pip install fastapi uvicorn

# Start server
uvicorn src.api.pricing_api:app --reload --port 8000
```

## Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test pricing recommendation
curl -X POST http://localhost:8000/api/v1/pricing/recommend \
  -H "Content-Type: application/json" \
  -d '{"merchant_id": "M123", "business_category": "E-commerce", ...}'
```
