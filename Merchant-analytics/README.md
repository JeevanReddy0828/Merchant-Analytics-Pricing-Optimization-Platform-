# Merchant Analytics & Pricing Optimization Platform

A production-grade data science platform for **merchant behavior analysis**, **pricing optimization**, and **transaction analytics** in the payments/fintech domain.

## 🎯 Project Overview

This platform demonstrates end-to-end data science capabilities for a two-sided payment network connecting merchants and consumers. It includes ML models for dynamic pricing, churn prediction, NLP-based feedback analysis, and automated reporting dashboards.

## 🚀 Key Features

| Component | Description |
|-----------|-------------|
| **Pricing Optimization** | Gradient Boosting model recommending optimal fee rates per merchant |
| **Churn Prediction** | Binary classifier identifying at-risk merchants with retention targeting |
| **Sentiment Analysis** | Hybrid NLP (rule-based + ML) for support ticket categorization |
| **SQL Analytics** | 9 production queries for dashboards and executive reporting |
| **Data Pipelines** | Batch ETL with feature engineering and aggregation stages |
| **API Serving** | FastAPI endpoints for real-time model predictions |

## 📁 Project Structure

```
merchant-analytics/
├── src/
│   ├── api/                # FastAPI model serving
│   │   └── pricing_api.py
│   ├── data/               # Data generation & SQL queries
│   │   ├── generator.py
│   │   ├── sql_queries.py
│   │   └── database.py
│   ├── models/             # ML models
│   │   ├── pricing_optimizer.py
│   │   ├── churn_predictor.py
│   │   └── sentiment_analyzer.py
│   ├── pipelines/          # Data pipelines
│   │   ├── batch_pipeline.py
│   │   └── feature_engineering.py
│   └── visualization/      # Dashboards & reports
│       └── dashboard_generator.py
├── tests/                  # pytest test suite
├── notebooks/              # Jupyter notebooks for EDA
├── scripts/                # CLI utilities
├── data/                   # Data storage
├── dashboards/             # Exported visualizations
├── docs/                   # Documentation
├── main.py                 # Full pipeline demo
├── requirements.txt        # Dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/merchant-analytics.git
cd merchant-analytics

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

### Option 1: Run Full Pipeline (Recommended)

This generates data, trains all models, and creates dashboards:

```bash
python main.py
```

**Output:**
- Synthetic data in `data/raw/`
- SQL queries in `dashboards/sql_queries.sql`
- Tableau exports in `dashboards/tableau_exports/`
- Monthly report in `dashboards/reports/`

### Option 2: Run Individual Components

#### Generate Synthetic Data
```bash
python scripts/generate_data.py --merchants 5000 --transactions 200000 --output data/raw
```

#### Train Models
```bash
# Train all models
python scripts/train_models.py --model all --data data/raw

# Train specific model
python scripts/train_models.py --model pricing --data data/raw
python scripts/train_models.py --model churn --data data/raw
```

#### Run Batch Pipeline
```bash
python -m src.pipelines.batch_pipeline
```

### Option 3: Start API Server

```bash
# Install FastAPI dependencies
pip install fastapi uvicorn

# Start server
uvicorn src.api.pricing_api:app --reload --port 8000

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 4: Run Jupyter Notebooks

```bash
pip install jupyter
jupyter notebook notebooks/
```

Available notebooks:
- `01_exploratory_data_analysis.ipynb` - Data exploration and visualization
- `02_model_training.ipynb` - Model training workflow

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html

# Run specific test class
pytest tests/test_analytics.py::TestPricingOptimizer -v
```

## 📊 Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Pricing Optimizer | R² | 0.99 |
| Pricing Optimizer | MAE | 0.0001 |
| Churn Predictor | AUC-ROC | 0.99 |
| Churn Predictor | F1 Score | 0.99 |
| Sentiment Analyzer | Accuracy | 1.00 |

## 📈 Sample Outputs

### Pricing Recommendations
```
merchant_id   pricing_tier  current_rate  recommended_rate  revenue_impact
MFC490CA45C00   enterprise         0.022            0.0248        899.52
M34ED066DF378   enterprise         0.022            0.0252        444.25
M4F6FFE13A5D7   enterprise         0.022            0.0252        428.53
```

### Churn Risk Tiers
```
Risk Tier Distribution:
  LOW       : 2,030 (67.7%)
  CRITICAL  :   970 (32.3%)

Revenue at Risk:
  CRITICAL: $37,003.47
  LOW: $10.87
```

### Monthly Executive Summary
```
Report Period: 2024-12
  Total Transactions: 4,186
  Total Volume: $301,415.54
  Total Fees: $9,337.47
  Active Merchants: 1,362
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/pricing/recommend` | POST | Get pricing recommendation |
| `/api/v1/pricing/batch` | POST | Batch pricing recommendations |
| `/api/v1/churn/assess` | POST | Assess merchant churn risk |
| `/api/v1/model/metrics` | GET | Get model performance metrics |

## 🛠️ Tech Stack

- **Languages:** Python 3.10+, SQL
- **ML/Stats:** Scikit-learn, XGBoost, Gradient Boosting
- **Data:** Pandas, NumPy, SQLite
- **Visualization:** Plotly, Matplotlib, Seaborn
- **API:** FastAPI, Uvicorn
- **Testing:** pytest, pytest-cov
- **Notebooks:** Jupyter

## 📚 Documentation

- [Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md) - Architecture and components
- [API Documentation](docs/API_DOCUMENTATION.md) - REST API reference

## 📝 License

MIT License

---

**Author:** Jeevan | Data Science Portfolio Project