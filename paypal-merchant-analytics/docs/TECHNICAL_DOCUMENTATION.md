# PayPal Merchant Analytics - Technical Documentation

## Project Overview

This project implements a production-grade data science platform for merchant analytics, pricing optimization, and churn prediction. It demonstrates skills directly aligned with PayPal's Data Scientist 1 role requirements.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Merchants │  │  Users   │  │Transactions│ │ Feedback │        │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼──────────────┼─────────────┼──────────────┘
        │             │              │             │
        ▼             ▼              ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Batch Pipeline                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │Ingestion │→ │Transform │→ │ Feature  │→ │Aggregation│        │
│  └──────────┘  └──────────┘  │Engineering│  └──────────┘        │
│                              └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML Models                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │    Pricing     │  │     Churn      │  │   Sentiment    │    │
│  │   Optimizer    │  │   Predictor    │  │   Analyzer     │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Layer                                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │  Tableau   │  │    API     │  │  Reports   │                │
│  │  Exports   │  │  Serving   │  │ Dashboard  │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Generation (`src/data/generator.py`)

Generates realistic PayPal-like synthetic data:
- **Merchants**: Business profiles with categories, regions, pricing tiers
- **Users**: Consumer profiles with payment preferences
- **Transactions**: Transaction records with fees, status, cross-border flags
- **Feedback**: Merchant support tickets for NLP analysis

### 2. SQL Analytics (`src/data/sql_queries.py`)

9 production-ready SQL queries for:
- Monthly transaction volume
- Merchant performance scorecard
- Pricing tier analysis
- Cross-border flow analysis
- Hourly transaction patterns
- Category performance
- Churn risk cohorts
- Platform performance
- Fee optimization analysis

### 3. ML Models

#### Pricing Optimizer (`src/models/pricing_optimizer.py`)
- **Algorithm**: Gradient Boosting Regressor
- **Features**: Volume, churn risk, cross-border ratio, etc.
- **Output**: Optimal fee rate recommendations
- **Metrics**: MAE, RMSE, R²

#### Churn Predictor (`src/models/churn_predictor.py`)
- **Algorithm**: Gradient Boosting Classifier
- **Features**: Recency, frequency, volume trends
- **Output**: Churn probability, risk tier
- **Metrics**: Accuracy, Precision, Recall, AUC-ROC

#### Sentiment Analyzer (`src/models/sentiment_analyzer.py`)
- **Approaches**: Rule-based + ML hybrid
- **Features**: TF-IDF with n-grams
- **Output**: Sentiment label, category, confidence

### 4. Data Pipelines (`src/pipelines/`)

- **Batch Pipeline**: Daily processing for analytics
- **Feature Engineering**: Centralized feature transformations

### 5. Visualization (`src/visualization/dashboard_generator.py`)

- Tableau-compatible CSV exports
- Interactive Plotly dashboards
- Automated monthly reports

### 6. API (`src/api/pricing_api.py`)

FastAPI service for real-time predictions:
- `POST /api/v1/pricing/recommend` - Get pricing recommendation
- `POST /api/v1/churn/assess` - Assess churn risk
- `GET /health` - Health check

## Usage

### Generate Data
```bash
python scripts/generate_data.py --merchants 5000 --transactions 200000
```

### Train Models
```bash
python scripts/train_models.py --model all --data data/raw
```

### Run Full Pipeline
```bash
python main.py
```

### Start API Server
```bash
uvicorn src.api.pricing_api:app --reload
```

### Run Tests
```bash
pytest tests/ -v --cov=src
```

## Key Metrics

| Model | Metric | Value |
|-------|--------|-------|
| Pricing | R² | 0.998 |
| Pricing | MAE | 0.0001 |
| Churn | AUC-ROC | 1.0 |
| Churn | F1 | 1.0 |
| Sentiment | Accuracy | 1.0 |

## Dependencies

- Python 3.10+
- pandas, numpy
- scikit-learn
- plotly
- FastAPI (optional)

## Author

**Jeevan** - Data Scientist Portfolio Project
Target Role: PayPal Data Scientist 1
