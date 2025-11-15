# backend/main.py (updated)
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import time
from api.models.database import init_db
import os
from datetime import datetime
from api.routes import upload, analysis, training, bias, models, dashboard, websocket_monitor, ab_testing, auth, audit, report, monitor, analyze, ai_agent
from core.src.logger import logging
from typing import AsyncGenerator
import asyncio


# Create FastAPI app with enhanced configuration
async def lifespan(app: FastAPI):
    logging.info("Starting BiasGuard API...")
    
    # Retry database initialization
    max_retries = 5
    for attempt in range(max_retries):
        try:
            init_db()
            logging.info("Database initialized")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"DB init attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2)
            else:
                logging.error(f"DB init failed after {max_retries} attempts: {e}")
                raise
    
    yield
    
    logging.info("Shutting down...")

app = FastAPI(
    title="Fairness & Bias Detection API",
    description="Comprehensive API for detecting fairness and bias in ML models with enterprise-grade features",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json", 
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001", # Streamlit default port
        "http://localhost:5173",
        "*",
        "ws://localhost:3000",
        "http://localhost:3000",
        # "http://localhost:3000",  # React dev server
        # "http://localhost:8080",  # Vue dev server
        # "https://your-production-domain.com"  # Add production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
#Version 1 API routes
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(training.router, prefix="/api/v1", tags=["training"])
app.include_router(bias.router, prefix="/api/v1", tags=["bias"])
# app.include_router(report.router, prefix="/reports", tags=["reports"])  # Commented out until report.py is implemented

#Version 2 API routes
app.include_router(auth.router, prefix="/api/v1", tags=["0. Authentication"])
app.include_router(models.router, prefix="/api/v1", tags=["1. Model Registry"])
app.include_router(monitor.router, prefix="/api/v1", tags=["2. Prediction Logging"])
app.include_router(analyze.router, prefix="/api/v1", tags=["3. Bias Analysis"])
app.include_router(report.router, prefix="/api/v1", tags=["4. Reports"])
app.include_router(websocket_monitor.router, prefix="/ws", tags=["5. Real-Time Monitoring"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["6. Dashboard"])
app.include_router(ab_testing.router, prefix="/api/v1", tags=["7. Model Comparison"])
app.include_router(audit.router, prefix="/api/v1", tags=["8. Audit Logs"])
app.include_router(ai_agent.router, prefix="/api/v1", tags=["9. AI Compliance Agent"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.2.0"}

@app.get("/")
async def root():
    return {
        "message": "Fairness & Bias Detection API",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/v1/upload",
            "analysis": "/api/v1/analysis", 
            "training": "/api/v1/training",
            "bias": "/api/v1/bias",
            "models": "/api/v1/models",
            "dashboard": "/api/v1/dashboard",
            
            "websocket_monitor": "/ws",
            "ab_testing": "/api/v1/ab_testing",
            "auth": "api/v1/auth",
            "report": "/api/v1/reports",
            "monitor": "/api/v1/monitoring",
            "analyze": "/api/v1/analyze"
        },
        "features": [
            "7 Fairness Metrics (Statistical Parity, Disparate Impact, etc.)",
            "AIF360 Integration",
            "3 Mitigation Strategies",
            "MLflow Experiment Tracking",
            "Real Time Model Comparison (A/B Testing)",
            "Real-time Bias Detection",
            "CFPB Compliance Reporting"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main.app", host="0.0.0.0", port=8001, reload=True)