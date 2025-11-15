
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import os


# Database URL Configuration
# Supports both SQLite (dev) and PostgreSQL (production)
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./biasguard2.0.db"  # Fallback for local dev
)

print(f"🔍 DATABASE_URL from environment: {os.getenv('DATABASE_URL')}")
print(f"🔍 SQLALCHEMY_DATABASE_URL being used: {SQLALCHEMY_DATABASE_URL}")
import sys; sys.stdout.flush()

# PostgreSQL URL Conversion (for Railway/Heroku)
# Some platforms use postgres:// instead of postgresql://
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)


# Engine Configuration
# Different connect_args for SQLite vs PostgreSQL
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False}  # SQLite-specific
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,  # Connection pool size
        max_overflow=10,  # Max connections beyond pool_size
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False  # Set to True for SQL logging (debug)
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(String, primary_key=True, index=True)
    organization_id = Column(String, nullable=True)
    created_by = Column(String, nullable=True)
    model_type = Column(String)
    task_type = Column(String)
    dataset_name = Column(String)
    target_column = Column(String)
    sensitive_columns = Column(JSON)
    feature_count = Column(Integer)
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    accuracy = Column(Float)
    bias_status = Column(String, default="unknown")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    mlflow_run_id = Column(String, index=True)

class BiasAnalysis(Base):
    __tablename__ = "bias_analyses"
    
    analysis_id = Column(String, primary_key=True, index=True)
    organization_id = Column(String, nullable=True)
    model_id = Column(String,index=True)
    compliance_status = Column(String)
    bias_status = Column(String)
    fairness_metrics = Column(JSON)
    aif360_metrics = Column(JSON, nullable=True)
    recommendations = Column(JSON)
    analyzed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    mlflow_run_id = Column(String, index=True)

class Mitigation(Base):
    __tablename__ = "mitigations"
    
    mitigation_id = Column(String, primary_key=True, index=True)
    original_model_id = Column(String, ForeignKey("models.model_id"))
    new_model_id = Column(String, ForeignKey("models.model_id"))
    strategy = Column(String)
    original_accuracy = Column(Float)
    new_accuracy = Column(Float)
    original_disparate_impact = Column(Float, nullable=True)
    new_disparate_impact = Column(Float, nullable=True)
    accuracy_impact = Column(Float)
    bias_improvement = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    mlflow_run_id = Column(String, index=True)

class ExternalModel(Base):
    __tablename__ = "external_models"
    
    model_id = Column(String, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String, nullable=True)
    framework = Column(String, nullable=True)
    version = Column(String, nullable=True)
    endpoint_url = Column(String, nullable=True)
    sensitive_attributes = Column(JSON, nullable=False)
    monitoring_enabled = Column(Boolean, default=True)
    alert_thresholds = Column(JSON, nullable=True)
    status = Column(String, default="active")
    organization_id = Column(String, index=True, nullable=True)
    created_by = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    prediction_logs = relationship("PredictionLog", back_populates="external_model")

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    log_id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("external_models.model_id"), index=True)
    prediction = Column(Integer)
    prediction_proba = Column(Float, nullable=True)
    ground_truth = Column(Integer, nullable=True)
    sensitive_attributes = Column(JSON, nullable=False)
    features = Column(JSON, nullable=True)
    organization_id = Column(String, index=True, nullable=True)
    logged_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    batch_id = Column(String, nullable=True, index=True)
    data_source = Column(String, nullable=True)
    
    external_model = relationship("ExternalModel", back_populates="prediction_logs")


def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {SQLALCHEMY_DATABASE_URL.split('@')[-1] if '@' in SQLALCHEMY_DATABASE_URL else SQLALCHEMY_DATABASE_URL}")

def get_db():
    """Dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()