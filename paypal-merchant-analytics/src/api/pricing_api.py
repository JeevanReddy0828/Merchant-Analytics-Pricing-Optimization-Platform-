"""
Pricing Optimization API

FastAPI service for real-time pricing recommendations.
Serves ML model predictions via REST endpoints.

Aligns with JD: Production quality code for data model pipelines
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

app = FastAPI(
    title="PayPal Pricing Optimization API",
    description="ML-powered dynamic pricing recommendations for merchants",
    version="1.0.0"
)

# Global model instance (loaded on startup)
pricing_model = None
churn_model = None


class MerchantInput(BaseModel):
    """Input schema for merchant pricing request."""
    merchant_id: str = Field(..., description="Unique merchant identifier")
    business_category: str = Field(..., description="Business vertical")
    region: str = Field(..., description="Geographic region")
    pricing_tier: str = Field(..., description="Current pricing tier")
    monthly_volume_usd: float = Field(..., ge=0, description="Monthly transaction volume")
    avg_transaction_size: float = Field(..., ge=0, description="Average transaction amount")
    account_age_months: int = Field(..., ge=0, description="Account age in months")
    cross_border_ratio: float = Field(0.0, ge=0, le=1, description="Cross-border transaction ratio")
    dispute_rate: float = Field(0.0, ge=0, le=1, description="Dispute rate")
    refund_rate: float = Field(0.0, ge=0, le=1, description="Refund rate")
    churn_risk_score: float = Field(0.0, ge=0, le=1, description="Churn risk score")


class PricingResponse(BaseModel):
    """Response schema for pricing recommendation."""
    merchant_id: str
    current_tier: str
    recommended_rate: float
    current_rate: float
    rate_change: float
    confidence: float
    factors: Dict[str, float]


class ChurnRiskResponse(BaseModel):
    """Response schema for churn risk assessment."""
    merchant_id: str
    churn_probability: float
    risk_tier: str
    revenue_at_risk: float
    recommended_action: str


class BatchPricingRequest(BaseModel):
    """Batch pricing request."""
    merchants: List[MerchantInput]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    version: str


@app.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global pricing_model, churn_model
    
    try:
        from pricing_optimizer import PricingOptimizer
        from churn_predictor import ChurnPredictor
        
        # In production, load pre-trained models from disk
        # For demo, models need to be trained first
        pricing_model = PricingOptimizer()
        churn_model = ChurnPredictor()
        
        print("Models initialized (require training before use)")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=pricing_model is not None,
        version="1.0.0"
    )


@app.post("/api/v1/pricing/recommend", response_model=PricingResponse)
async def get_pricing_recommendation(merchant: MerchantInput):
    """
    Get optimal pricing recommendation for a single merchant.
    
    Returns recommended fee rate based on merchant characteristics
    and predicted behavior patterns.
    """
    if pricing_model is None or not pricing_model.is_fitted:
        raise HTTPException(
            status_code=503, 
            detail="Pricing model not trained. Train model first."
        )
    
    import pandas as pd
    
    # Convert to DataFrame for model
    merchant_df = pd.DataFrame([merchant.dict()])
    
    # Get prediction
    recommended_rate = pricing_model.predict(merchant_df)[0]
    
    # Get current rate based on tier
    tier_rates = {
        'standard': 0.029,
        'preferred': 0.025,
        'enterprise': 0.022,
        'custom': 0.020
    }
    current_rate = tier_rates.get(merchant.pricing_tier, 0.029)
    
    return PricingResponse(
        merchant_id=merchant.merchant_id,
        current_tier=merchant.pricing_tier,
        recommended_rate=round(recommended_rate, 4),
        current_rate=current_rate,
        rate_change=round(recommended_rate - current_rate, 4),
        confidence=0.85,  # Placeholder
        factors={
            "volume_impact": -0.002 if merchant.monthly_volume_usd > 10000 else 0,
            "churn_risk_impact": -merchant.churn_risk_score * 0.003,
            "cross_border_impact": merchant.cross_border_ratio * 0.01
        }
    )


@app.post("/api/v1/pricing/batch", response_model=List[PricingResponse])
async def batch_pricing_recommendations(request: BatchPricingRequest):
    """Get pricing recommendations for multiple merchants."""
    results = []
    for merchant in request.merchants:
        result = await get_pricing_recommendation(merchant)
        results.append(result)
    return results


@app.post("/api/v1/churn/assess", response_model=ChurnRiskResponse)
async def assess_churn_risk(merchant: MerchantInput):
    """
    Assess churn risk for a merchant.
    
    Returns probability of churn and recommended retention actions.
    """
    if churn_model is None or not churn_model.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Churn model not trained. Train model first."
        )
    
    import pandas as pd
    
    merchant_df = pd.DataFrame([merchant.dict()])
    
    # Get prediction
    churn_prob = churn_model.predict_proba(merchant_df)[0]
    
    # Determine risk tier
    if churn_prob >= 0.7:
        risk_tier = "CRITICAL"
        action = "Immediate account manager outreach + pricing review"
    elif churn_prob >= 0.5:
        risk_tier = "HIGH"
        action = "Automated re-engagement campaign"
    elif churn_prob >= 0.3:
        risk_tier = "MEDIUM"
        action = "Feature adoption email series"
    else:
        risk_tier = "LOW"
        action = "Standard engagement"
    
    # Calculate revenue at risk
    monthly_revenue = merchant.monthly_volume_usd * 0.025
    revenue_at_risk = monthly_revenue * churn_prob
    
    return ChurnRiskResponse(
        merchant_id=merchant.merchant_id,
        churn_probability=round(churn_prob, 4),
        risk_tier=risk_tier,
        revenue_at_risk=round(revenue_at_risk, 2),
        recommended_action=action
    )


@app.get("/api/v1/model/metrics")
async def get_model_metrics():
    """Get current model performance metrics."""
    return {
        "pricing_model": {
            "status": "trained" if pricing_model and pricing_model.is_fitted else "not_trained",
            "features": pricing_model.feature_names if pricing_model and pricing_model.is_fitted else []
        },
        "churn_model": {
            "status": "trained" if churn_model and churn_model.is_fitted else "not_trained",
            "features": churn_model.feature_names if churn_model and churn_model.is_fitted else []
        }
    }


# Run with: uvicorn pricing_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
