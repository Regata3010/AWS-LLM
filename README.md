# BiasGuard — Production AI Fairness Monitoring Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19+-blue.svg)](https://reactjs.org)
[![Deployed on GKE](https://img.shields.io/badge/Deployed-GKE-blue.svg)](https://cloud.google.com/kubernetes-engine)

BiasGuard monitors AI models already deployed in production to ensure CFPB, EEOC, and EU AI Act compliance. Built for FinTech BNPL companies, lending platforms, and HR tech to detect bias before regulators do.

**Stop training models. Start monitoring the ones you already have.**

---

## Problem

Companies have ML models already deployed — trained by their data science teams, running in production, making decisions that affect real people. But they have no visibility into whether those models are discriminating against protected classes until they get sued or fined.

- CFPB fines: $1M - $100M+ for biased lending models
- Class-action discrimination suits cost millions in legal fees
- Models pulled from production cause direct revenue loss

BiasGuard monitors production models 24/7 and surfaces violations before regulators do.

---

## Architecture

```mermaid
flowchart TD
    A([External Model<br/>SageMaker / Azure ML / Databricks]) -->|POST predictions + sensitive attributes| B

    B[Prediction Logging API<br/>1000+ predictions per second]
    B --> C

    C[Bias Detection Engine<br/>core/bias_engine.py · AIF360]
    C --> D[7 Fairness Metrics<br/>Statistical Parity · Disparate Impact<br/>Equal Opportunity · Average Odds<br/>Theil Index · Generalized Entropy · CV]
    D --> E

    E[Intersectionality Analysis<br/>core/intersectionality.py<br/>race + gender · age + disability]
    E --> F{Violation<br/>Detected?}

    F -->|Yes| G[Alert Engine<br/>Slack · Email · Webhook]
    F -->|No| H[Dashboard Update<br/>Real-time via WebSocket]

    G --> I
    H --> I

    I[AI Compliance Agent<br/>agent/compliance_agent.py<br/>LangGraph + GPT-4]
    I --> J[RAG<br/>CFPB/EEOC regulations<br/>Pinecone vector store]
    I --> K[CAG<br/>Real-time model data<br/>live prediction context]

    J & K --> L

    L[Compliance Report Generation<br/>agent/report_generator.py]
    L --> M([CFPB Adverse Action Notices<br/>EEOC Audit Documentation<br/>Executive Summaries<br/>Complete Audit Trail])

    style A fill:#6366f1,color:#fff
    style M fill:#374151,color:#fff
    style F fill:#d97706,color:#fff
    style G fill:#dc2626,color:#fff
    style H fill:#059669,color:#fff
    style I fill:#7c3aed,color:#fff
```

---

## How It Works

**1. Register your model** — point BiasGuard at any model running on SageMaker, Azure ML, Databricks, or your own infrastructure.

**2. Log predictions** — send predictions and sensitive attributes to the BiasGuard API as your model runs. Handles 1000+ predictions per second.

**3. Bias detection runs automatically** — AIF360 computes 7 fairness metrics including intersectionality analysis across protected attribute combinations.

**4. Violations trigger alerts** — Slack, email, or webhook notifications when thresholds are breached.

**5. AI compliance agent answers questions** — LangGraph-based agent with RAG over CFPB/EEOC regulation knowledge base answers natural language compliance queries against your live model data.

**6. Reports generated on demand** — CFPB adverse action notices, EEOC statistical evidence, executive summaries.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Bias Detection | AIF360 (IBM), 7 fairness metrics |
| AI Compliance Agent | LangGraph, GPT-4, Pinecone (RAG) |
| Backend | FastAPI, SQLAlchemy, PostgreSQL, Redis |
| Frontend | React 19, TypeScript, Tailwind CSS, Recharts |
| Real-time | WebSockets |
| Infrastructure | Docker, Kubernetes (GKE), Nginx |
| Model Registry | External — SageMaker, Azure ML, Databricks |

---

## Fairness Metrics

| Metric | Description | Compliant Range | Regulation |
|---|---|---|---|
| Disparate Impact | Ratio of favorable outcomes | [0.8, 1.25] | EEOC Four-Fifths Rule |
| Statistical Parity | Difference in approval rates | [-0.1, 0.1] | CFPB Guidance |
| Equal Opportunity | True positive rate parity | [-0.1, 0.1] | EEOC Title VII |
| Average Odds | TPR and FPR parity | [-0.1, 0.1] | EU AI Act |
| Theil Index | Income inequality analogue for predictions | < 0.1 | Internal threshold |
| Generalized Entropy | Distribution-level fairness | < 0.1 | Internal threshold |
| Coefficient of Variation | Spread of outcomes across groups | < 0.1 | Internal threshold |

---

## Key Features

**External Model Registry** — monitor models deployed anywhere, no retraining or redeployment required.

**Intersectionality Analysis** — detects compound bias across multiple protected attributes simultaneously (race + gender, age + disability).

**AI Compliance Agent** — LangGraph-based agent combines RAG over CFPB/EEOC regulations with real-time access to your model data. Example queries:
- "Is my loan approval model compliant with ECOA?"
- "What is the four-fifths rule and do I pass it?"
- "Which models have violations this month?"
- "Explain Regulation B Section 1002.6(a)"

**Automated Compliance Reporting** — CFPB adverse action notices, EEOC audit documentation, LLM-generated executive summaries, complete audit trail.

**Note on V1.0** — BiasGuard V1.0 included a training platform where users could upload CSVs, train models with LLM-assisted column detection, and apply bias mitigation. That architecture was deprecated in V2.0 in favor of monitoring-first design.

---

## Quick Start

**Prerequisites:** Python 3.10+, Node.js 20+, PostgreSQL 14+, Redis 7+, Docker

```bash
git clone https://github.com/Regata3010/biasguard.git
cd biasguard

cp deployment/.env.example deployment/.env
# Edit deployment/.env with your API keys

docker-compose -f deployment/docker-compose.yml up -d
```

Access at `http://localhost`. API docs at `http://localhost:8001/docs`.

**Manual setup:**

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8001

# Frontend
cd frontend
npm install
npm run dev
```

---

## Environment Variables

**Backend:**

```bash
DATABASE_URL=postgresql://user:password@localhost:5432/biasguard
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east1-gcp
SECRET_KEY=your-secret-key-32-chars-min
```

**Frontend:**

```bash
VITE_API_URL=http://localhost:8001/api/v1
```

---

## API Examples

**Register a model:**

```bash
POST /api/v1/models/register
{
  "model_name": "Loan Approval Model",
  "model_type": "classification",
  "sensitive_attributes": ["race", "gender", "age"]
}
```

**Log predictions:**

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

**Run bias analysis:**

```bash
POST /api/v1/analyze
{
  "model_id": "model_abc123",
  "period_days": 30
}
```

**Query the compliance agent:**

```bash
POST /api/v1/agent/chat
{
  "message": "Is my loan model compliant with ECOA?"
}
```

---

## Project Structure

```
biasguard/
├── backend/
│   ├── main.py                      # FastAPI entry point
│   ├── api/
│   │   ├── models.py                # Model registry endpoints
│   │   ├── monitor.py               # Prediction logging endpoints
│   │   ├── analyze.py               # Bias analysis endpoints
│   │   └── agent.py                 # AI compliance agent endpoints
│   ├── core/
│   │   ├── bias_engine.py           # AIF360 fairness computation
│   │   ├── intersectionality.py     # Cross-attribute bias detection
│   │   ├── drift_detection.py       # Data and fairness drift
│   │   └── alert_engine.py          # Violation alerting
│   ├── agent/
│   │   ├── compliance_agent.py      # LangGraph agent definition
│   │   ├── rag_retriever.py         # CFPB/EEOC regulation RAG
│   │   └── report_generator.py      # Automated compliance reports
│   └── models/
│       └── schemas.py               # SQLAlchemy models
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Dashboard.tsx        # Real-time monitoring
│       │   ├── AgentChat.tsx        # AI compliance chat
│       │   └── ReportGenerator.tsx  # Report UI
│       └── hooks/
│           └── useWebSocket.ts      # Real-time updates
└── deployment/
    ├── docker-compose.yml
    ├── .env.example
    └── k8s/                         # GKE manifests
```

---

## Deployment

**Local:**

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

**Production (GKE):**

```bash
kubectl apply -f deployment/k8s/
```

---

## Use Cases

**FinTech / BNPL Lending** — monitor loan approval models for ECOA compliance, detect racial and gender bias in credit decisions.

**HR Tech / Recruiting** — monitor resume screening models for EEOC compliance, ensure diverse candidate selection.

**Healthcare** — monitor patient risk models for equitable care outcomes across demographic groups.

---

## Contact

**Arav Pandey**
- GitHub: [@Regata3010](https://github.com/Regata3010)
- LinkedIn: [linkedin.com/in/aravpandey](https://www.linkedin.com/in/aravpandey/)
- Email: nuclearreactor3010@gmail.com

---

## Acknowledgments

- [AIF360](https://github.com/Trusted-AI/AIF360) — IBM fairness toolkit
- [LangChain](https://www.langchain.com/) — LLM application framework
- [FastAPI](https://fastapi.tiangolo.com/) — Python web framework
- [Pinecone](https://www.pinecone.io/) — Vector database for RAG