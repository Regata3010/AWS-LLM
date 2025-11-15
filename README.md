# 🛡️ BiasGuard

**Production AI Fairness Monitoring & Compliance Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-blue.svg)](https://reactjs.org)
[![Deployed on GKE](https://img.shields.io/badge/Deployed-GKE-blue.svg)](https://cloud.google.com/kubernetes-engine)

BiasGuard monitors AI models **already deployed in production** to ensure CFPB, EEOC, and EU AI Act compliance. Built for FinTech BNPL companies, lending platforms, and HR tech to avoid regulatory fines and discrimination lawsuits.

**Stop training models. Start monitoring the ones you already have.** 🎯

---

## 🎯 **Key Features**

### **📊 Production Model Monitoring**
- **External Model Registry**: Monitor models deployed on SageMaker, Azure ML, Databricks, anywhere
- **Prediction Logging**: Capture 1000s+ predictions per second via API
- **Real-time Bias Detection**: WebSocket-based live monitoring dashboard
- **Drift Detection**: Track data drift, concept drift, and fairness drift over time

### **🔍 Enterprise Bias Detection**
- **7 Fairness Metrics**: Statistical Parity, Disparate Impact, Equal Opportunity, Average Odds, Theil Index, Generalized Entropy, Coefficient of Variation
- **AIF360 Integration**: Industry-standard fairness library from IBM
- **Intersectionality Analysis**: Detect compound bias across multiple protected attributes (race + gender, age + disability)
- **Historical Tracking**: Monitor bias trends over weeks/months

### **🤖 AI Compliance Agent**
- **LangGraph-based Agent**: Multi-tool autonomous reasoning with RAG + CAG
- **RAG (Retrieval-Augmented Generation)**: CFPB/EEOC regulation knowledge base in Pinecone
- **CAG (Context-Augmented Generation)**: Real-time access to your model data
- **Natural Language Queries**: 
  - "Is my loan approval model compliant with ECOA?"
  - "What's the four-fifths rule and do I pass it?"
  - "Which models have violations this month?"
  - "Explain Regulation B Section 1002.6(a)"

### **📄 Automated Compliance Reporting**
- **CFPB Adverse Action Notices**: Auto-generated documentation
- **EEOC Compliance Reports**: Statistical evidence for audits
- **Executive Summaries**: LLM-generated insights for leadership
- **Audit Trail**: Complete history of predictions, analyses, and violations

### **⚡ Real-Time Alerting**
- **Violation Alerts**: Instant notifications when bias thresholds exceeded
- **Slack/Email Integration**: Alert your compliance team immediately
- **Customizable Thresholds**: Set your own risk tolerance per model

### **🎓 Model Training Platform (Legacy V1.0)**
*Note: BiasGuard V1.0 includes a training platform where users can upload CSVs, train models with LLM-assisted column selection, and apply bias mitigation. This is available but not the primary focus—BiasGuard is now optimized for monitoring production models.*

---

## 💼 **Why BiasGuard?**

### **The Problem:**
Companies have ML models **already deployed** (trained by their data science teams). But they have **no idea** if these models are discriminating against protected classes—until they get sued or fined.

### **The Consequences:**
- **CFPB Fines**: $1M - $100M+ for biased lending models
- **Lawsuits**: Class-action discrimination suits cost millions
- **Reputation Damage**: Public scandals destroy brand trust
- **Lost Revenue**: Models get pulled from production

### **The Solution:**
**BiasGuard monitors your production models 24/7** and alerts you to violations **before** regulators find them.

---

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                   BiasGuard Platform (V2.0)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (React + TypeScript)                              │
│  ├─ Real-time Monitoring Dashboard (WebSockets)             │
│  ├─ AI Compliance Chat Interface                            │
│  ├─ External Model Registry                                  │
│  ├─ Prediction Logging UI                                    │
│  └─ Compliance Report Generator                              │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Backend (FastAPI + Python)                                 │
│  ├─ Bias Detection Engine (AIF360)                          │
│  │   └─ 7 fairness metrics, intersectionality analysis      │
│  ├─ AI Compliance Agent (LangGraph + GPT-4)                 │
│  │   ├─ RAG: CFPB/EEOC regulations (Pinecone)               │
│  │   └─ CAG: Real-time model data access                    │
│  ├─ Prediction Logging API (1000s+ predictions/sec)         │
│  ├─ Real-time Monitoring (WebSockets, drift detection)      │
│  ├─ Report Generation (CFPB, EEOC, executive summaries)     │
│  └─ Alert Engine (Slack, email, webhooks)                   │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Layer                                                  │
│  ├─ PostgreSQL (Model registry, predictions, audit logs)    │
│  ├─ Redis (Cache, real-time updates)                        │
│  ├─ Pinecone (CFPB/EEOC regulation vector store)            │
│  └─ S3/GCS (Reports, datasets)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Your Production Environment:
├─ SageMaker / Azure ML / Databricks models
│   └─ POST predictions to BiasGuard API
└─ BiasGuard monitors → Detects bias → Alerts you
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.10+
- Node.js 20+
- PostgreSQL 14+
- Redis 7+
- Docker (optional)

### **1. Clone Repository**
```bash
git clone https://github.com/Regata3010/biasguard.git
cd biasguard
```

### **2. Docker Deployment (Recommended)**
```bash
# Copy environment template
cp deployment/.env.example deployment/.env
# Edit deployment/.env with your API keys

# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# Access BiasGuard
open http://localhost
```

### **3. Manual Setup**

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your API keys
uvicorn main:app --reload --port 8001
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Access:**
- Dashboard: http://localhost:5173
- API Docs: http://localhost:8001/docs

---

## 🔑 **Environment Variables**

### **Backend**
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/biasguard
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east1-gcp
SECRET_KEY=your-secret-key-32-chars-min
```

### **Frontend**
```bash
VITE_API_URL=http://localhost:8001/api/v1
```

---

## 📖 **API Documentation**

Full interactive API documentation: http://localhost:8001/docs

### **Quick Examples:**

**Register Model:**
```bash
POST /api/v1/models/register
{
  "model_name": "Loan Approval Model",
  "model_type": "classification",
  "sensitive_attributes": ["race", "gender", "age"]
}
```

**Log Predictions:**
```bash
POST /api/v1/monitor/batch
{
  "model_id": "model_abc123",
  "predictions": [1, 0, 1],
  "sensitive_attributes": [
    {"race": "White", "gender": "Male", "age": 35},
    {"race": "Black", "gender": "Female", "age": 28}
  ]
}
```

**Analyze Bias:**
```bash
POST /api/v1/analyze
{
  "model_id": "model_abc123",
  "period_days": 30
}
```

**Chat with AI Agent:**
```bash
POST /api/v1/agent/chat
{
  "message": "Is my loan model compliant?"
}
```

---

## 📊 **Fairness Metrics**

| Metric | Description | Compliant Range | Regulation |
|--------|-------------|-----------------|------------|
| **Disparate Impact** | Ratio of favorable outcomes | [0.8, 1.25] | EEOC Four-Fifths Rule |
| **Statistical Parity** | Difference in approval rates | [-0.1, 0.1] | CFPB Guidance |
| **Equal Opportunity** | True positive rate parity | [-0.1, 0.1] | EEOC Title VII |
| **Average Odds** | TPR and FPR parity | [-0.1, 0.1] | EU AI Act |

---

## 🎯 **Use Cases**

### **FinTech / BNPL Lending**
Monitor loan approval models for ECOA compliance and detect racial/gender bias in credit decisions.

### **HR Tech / Recruiting**
Monitor resume screening models for EEOC compliance and ensure diverse candidate pools.

### **Healthcare**
Monitor patient risk models for equitable care and track health equity metrics.

---

## 🛠️ **Technology Stack**

**Backend:** FastAPI, SQLAlchemy, AIF360, LangChain, LangGraph, OpenAI GPT-4, Pinecone, PostgreSQL, Redis

**Frontend:** React 19, TypeScript, Tailwind CSS, Recharts, WebSockets

**Infrastructure:** Docker, Kubernetes (GKE), Nginx, Cloud SQL, Google Container Registry

---

## 🚀 **Deployment**

### **Local (Docker)**
```bash
docker-compose -f deployment/docker-compose.yml up -d
```

### **Production (GKE)**
```bash
kubectl apply -f deployment/k8s/
```

---

## 📝 **License**

This project is licensed under the MIT License.

---

## 🗺️ **Roadmap**

### **V2.0 (Current)**
- ✅ External model registry
- ✅ 7 fairness metrics with AI agent
- ✅ Real-time monitoring
- ✅ CFPB compliance reports

### **V2.1 (Q1 2026)**
- 🔲 Slack/Teams integration
- 🔲 Custom alert thresholds
- 🔲 Multi-region deployment
- 🔲 SSO (SAML/OAuth)

### **V3.0 (Q3 2026)**
- 🔲 Multi-tenancy for enterprises
- 🔲 SOC2 Type II certification
- 🔲 Global model marketplace

---

## 📧 **Contact**

**Aarav Pandey**
- Email: nuclearreactor3010@gmail.com
- GitHub: [@Regata3010](https://github.com/Regata3010)
- LinkedIn: [linkedin.com/in/aravpandey](https://www.linkedin.com/in/aravpandey/)

---

## 🙏 **Acknowledgments**

- [AIF360](https://github.com/Trusted-AI/AIF360) - IBM's fairness toolkit
- [LangChain](https://www.langchain.com/) - LLM application framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Pinecone](https://www.pinecone.io/) - Vector database for RAG

---

<div align="center">

**Built with ❤️ for a fairer AI future**

**Monitor your models. Avoid lawsuits. Stay compliant.** 🛡️

[Report Bug](https://github.com/Regata3010/biasguard/issues) · [Request Feature](https://github.com/Regata3010/biasguard/issues)

</div>