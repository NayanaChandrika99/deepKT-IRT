# Production Deployment Guide for UWorld

This document outlines how to deploy the DeepKT + Wide&Deep IRT system into UWorld's production environment.

## Architecture Overview

### Current State (Demo)
- CLI-based (`scripts/demo_trace.py`)
- Reads pre-computed parquet files
- Batch processing (offline inference)
- No real-time API

### Target State (Production)
- REST API service
- Real-time inference on-demand
- Database-backed (not parquet files)
- Model serving infrastructure
- Batch jobs for model retraining

---

## Deployment Options

### Option 1: Microservices Architecture (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    UWorld Frontend                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Recommendation API Service                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  POST /api/v1/recommendations                         │  │
│  │  GET  /api/v1/mastery/{student_id}                    │  │
│  │  GET  /api/v1/explain/{student_id}/{skill}            │  │
│  │  POST /api/v1/gaming-check                            │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────┬───────────────────────────────┬─────────────┘
               │                                 │
               ▼                                 ▼
    ┌──────────────────┐            ┌──────────────────────┐
    │  SAKT Inference  │            │  WD-IRT Inference     │
    │  Service          │            │  Service              │
    │  (PyTorch)        │            │  (PyTorch Lightning)  │
    └──────────────────┘            └──────────────────────┘
               │                                 │
               └──────────────┬──────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Model Registry  │
                    │  (S3/GCS)        │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  UWorld Database │
                    │  (PostgreSQL)    │
                    └──────────────────┘
```

**Components:**
1. **Recommendation API** (FastAPI/Flask) - Main entry point
2. **SAKT Inference Service** - Loads SAKT checkpoint, runs inference
3. **WD-IRT Inference Service** - Loads WD-IRT checkpoint, runs inference
4. **Model Registry** - Stores checkpoints (S3, GCS, or UWorld's storage)
5. **Database** - Stores events, predictions, recommendations

---

### Option 2: Monolithic API (Simpler, Faster to Deploy)

Single FastAPI service that:
- Loads both models at startup
- Handles all inference internally
- Connects to UWorld database
- Simpler deployment, but less scalable

---

## Step-by-Step Deployment Plan

### Phase 1: API Service (Week 1-2)

#### 1.1 Create FastAPI Service

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="DeepKT Recommendation API")

class RecommendationRequest(BaseModel):
    student_id: str
    skill: str
    time_window: Optional[str] = None
    max_items: int = 5
    use_rl: bool = False

class RecommendationResponse(BaseModel):
    student_id: str
    skill: str
    mastery: float
    recommendations: List[dict]
    item_health_flags: List[dict]

@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized item recommendations for a student.
    
    This is the main production endpoint that replaces demo_trace.py.
    """
    # Load student mastery (from DB or cache)
    mastery = await load_student_mastery(request.student_id, request.skill)
    
    # Load item parameters (from DB or cache)
    item_params = await load_item_params(request.skill)
    
    # Generate recommendations
    if request.use_rl:
        recs = await recommend_items_rl(
            student_id=request.student_id,
            skill=request.skill,
            mastery=mastery,
            item_params=item_params,
            max_items=request.max_items
        )
    else:
        recs = await recommend_items(
            student_id=request.student_id,
            skill=request.skill,
            mastery=mastery,
            item_params=item_params,
            max_items=request.max_items
        )
    
    # Check item health
    health_flags = await check_item_health([r.item_id for r in recs])
    
    return RecommendationResponse(
        student_id=request.student_id,
        skill=request.skill,
        mastery=mastery,
        recommendations=[r.dict() for r in recs],
        item_health_flags=health_flags
    )

@app.get("/api/v1/mastery/{student_id}")
async def get_mastery(student_id: str, skill: Optional[str] = None):
    """Get student mastery scores."""
    pass

@app.get("/api/v1/explain/{student_id}/{skill}")
async def explain_mastery(student_id: str, skill: str):
    """Generate explanation for mastery score."""
    pass

@app.post("/api/v1/gaming-check")
async def check_gaming(student_id: str):
    """Check for gaming behavior."""
    pass
```

#### 1.2 Model Loading Service

```python
# src/api/model_loader.py
import torch
from pathlib import Path
from typing import Optional

class ModelRegistry:
    """Manages loading and caching of trained models."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self._sakt_model = None
        self._wdirt_model = None
    
    def load_sakt(self, checkpoint_name: str):
        """Load SAKT model (lazy, cached)."""
        if self._sakt_model is None:
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            self._sakt_model = torch.load(checkpoint_path, map_location="cpu")
            self._sakt_model.eval()
        return self._sakt_model
    
    def load_wdirt(self, checkpoint_name: str):
        """Load WD-IRT model (lazy, cached)."""
        if self._wdirt_model is None:
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            self._wdirt_model = torch.load(checkpoint_path, map_location="cpu")
            self._wdirt_model.eval()
        return self._wdirt_model
```

---

### Phase 2: Database Integration (Week 2-3)

#### 2.1 Schema Design

```sql
-- UWorld database schema additions

-- Student mastery (from SAKT)
CREATE TABLE student_mastery (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(255) NOT NULL,
    skill_id VARCHAR(50) NOT NULL,
    mastery_score FLOAT NOT NULL,
    interaction_count INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(student_id, skill_id)
);

CREATE INDEX idx_student_mastery_student ON student_mastery(student_id);
CREATE INDEX idx_student_mastery_skill ON student_mastery(skill_id);

-- Item parameters (from WD-IRT)
CREATE TABLE item_parameters (
    id SERIAL PRIMARY KEY,
    item_id VARCHAR(255) NOT NULL UNIQUE,
    skill_id VARCHAR(50) NOT NULL,
    difficulty FLOAT NOT NULL,
    discrimination FLOAT NOT NULL,
    guessing FLOAT NOT NULL,
    drift_score FLOAT,
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_item_params_skill ON item_parameters(skill_id);

-- Recommendations (audit trail)
CREATE TABLE recommendations (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(255) NOT NULL,
    skill_id VARCHAR(50) NOT NULL,
    item_id VARCHAR(255) NOT NULL,
    recommendation_type VARCHAR(20) NOT NULL, -- 'rule-based' or 'rl'
    expected_reward FLOAT,
    uncertainty FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- RL bandit state
CREATE TABLE bandit_state (
    id SERIAL PRIMARY KEY,
    state_data BYTEA NOT NULL, -- Serialized numpy arrays
    last_updated TIMESTAMP DEFAULT NOW()
);
```

#### 2.2 Data Pipeline Integration

```python
# src/api/data_loader.py
import asyncpg
from typing import Optional

class DatabaseLoader:
    """Loads data from UWorld database instead of parquet files."""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool
    
    async def load_student_mastery(
        self, 
        student_id: str, 
        skill_id: str
    ) -> Optional[float]:
        """Load mastery from database."""
        query = """
            SELECT mastery_score 
            FROM student_mastery 
            WHERE student_id = $1 AND skill_id = $2
        """
        row = await self.db.fetchrow(query, student_id, skill_id)
        return row['mastery_score'] if row else None
    
    async def load_item_params(self, skill_id: str) -> List[dict]:
        """Load item parameters for a skill."""
        query = """
            SELECT item_id, difficulty, discrimination, guessing, drift_score
            FROM item_parameters
            WHERE skill_id = $1
        """
        rows = await self.db.fetch(query, skill_id)
        return [dict(row) for row in rows]
```

---

### Phase 3: Batch Processing Pipeline (Week 3-4)

#### 3.1 Model Retraining Job

```python
# src/api/jobs/retrain_models.py
"""
Scheduled job (daily/weekly) to retrain models on new data.
"""
import asyncio
from datetime import datetime

async def retrain_sakt():
    """Retrain SAKT model on latest data."""
    # 1. Export new events from UWorld DB
    # 2. Train SAKT model
    # 3. Export predictions to DB
    # 4. Update model registry
    pass

async def retrain_wdirt():
    """Retrain WD-IRT model on latest data."""
    # 1. Export new clickstream data
    # 2. Train WD-IRT model
    # 3. Export item parameters to DB
    # 4. Update model registry
    pass
```

#### 3.2 Mastery Aggregation Job

```python
# src/api/jobs/aggregate_mastery.py
"""
Scheduled job to aggregate per-interaction mastery → per-skill mastery.
Runs after SAKT inference completes.
"""
async def aggregate_all_mastery():
    """Aggregate mastery for all students."""
    # 1. Load per-interaction mastery from DB
    # 2. Aggregate by skill
    # 3. Write to student_mastery table
    pass
```

---

### Phase 4: Model Serving Infrastructure (Week 4-5)

#### 4.1 Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model checkpoints
COPY reports/checkpoints/ /app/models/

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Set environment variables
ENV MODEL_DIR=/app/models
ENV DATABASE_URL=${DATABASE_URL}

# Run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4.2 Kubernetes Deployment (Optional)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepkt-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: uworld/deepkt-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: deepkt-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: deepkt-api-service
spec:
  selector:
    app: deepkt-api
  ports:
  - port: 80
    targetPort: 8000
```

---

## Integration Points with UWorld

### 1. Event Ingestion

**Current:** Reads from parquet files  
**Production:** Stream events from UWorld's event bus

```python
# src/api/event_ingestion.py
async def ingest_learning_event(event: dict):
    """
    Called when UWorld emits a learning event.
    Updates student mastery in real-time.
    """
    # 1. Run SAKT inference on new event
    # 2. Update mastery cache/DB
    # 3. Trigger recommendation refresh if needed
    pass
```

### 2. Item Catalog Integration

**Current:** Reads item_params.parquet  
**Production:** Query UWorld's item catalog API

```python
# src/api/item_catalog.py
async def sync_item_parameters():
    """
    Sync item parameters from WD-IRT to UWorld's item catalog.
    Updates item difficulty/discrimination metadata.
    """
    pass
```

### 3. Recommendation Delivery

**Current:** CLI output  
**Production:** Push to UWorld's recommendation queue

```python
# src/api/recommendation_delivery.py
async def deliver_recommendation(student_id: str, recommendations: List[dict]):
    """
    Send recommendations to UWorld's recommendation service.
    """
    # Push to UWorld's internal queue/API
    pass
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Train models on UWorld's production data
- [ ] Export model checkpoints to model registry
- [ ] Set up database schema
- [ ] Create API service code
- [ ] Write integration tests
- [ ] Set up monitoring/observability

### Deployment
- [ ] Deploy API service (Docker/K8s)
- [ ] Configure database connections
- [ ] Set up batch jobs (retraining, aggregation)
- [ ] Configure model registry access
- [ ] Set up load balancing
- [ ] Configure auto-scaling

### Post-Deployment
- [ ] Monitor API latency (< 200ms p95)
- [ ] Monitor model inference time (< 100ms p95)
- [ ] Set up alerting for errors
- [ ] Track recommendation quality metrics
- [ ] A/B test RL vs rule-based recommendations

---

## Performance Targets

- **API Latency:** < 200ms (p95) for recommendations
- **Model Inference:** < 100ms (p95) per prediction
- **Throughput:** 1000+ requests/second
- **Availability:** 99.9% uptime
- **Model Freshness:** Retrain weekly

---

## Monitoring & Observability

### Metrics to Track
- API request rate, latency, error rate
- Model inference time, cache hit rate
- Recommendation click-through rate
- Student mastery accuracy (AUC)
- Item parameter drift alerts

### Logging
- All API requests (with student_id anonymized)
- Model inference calls
- Recommendation generation
- Gaming detection alerts

---

## Security Considerations

1. **Authentication:** Integrate with UWorld's auth system
2. **Authorization:** Role-based access (students vs educators)
3. **Data Privacy:** Anonymize student IDs in logs
4. **Model Security:** Secure model registry access
5. **Rate Limiting:** Prevent API abuse

---

## Cost Estimation

### Infrastructure Costs (Monthly)
- **API Service:** $200-500 (3-5 instances)
- **Database:** $100-300 (PostgreSQL)
- **Model Storage:** $50-100 (S3/GCS)
- **Batch Jobs:** $100-200 (compute for retraining)
- **Total:** ~$450-1100/month

### Development Costs
- **API Development:** 2-3 weeks (1 engineer)
- **Database Integration:** 1-2 weeks (1 engineer)
- **Testing & Deployment:** 1 week (1 engineer)
- **Total:** 4-6 weeks

---

## Next Steps

1. **Choose deployment option** (Microservices vs Monolithic)
2. **Set up development environment** (local API + test DB)
3. **Implement API endpoints** (start with recommendations)
4. **Integrate with UWorld database** (replace parquet reads)
5. **Deploy to staging** (test with real UWorld data)
6. **Production rollout** (gradual, with monitoring)

---

## Questions for UWorld

Before finalizing deployment:

1. **Database:** What database does UWorld use? (PostgreSQL, MySQL, etc.)
2. **Event Stream:** How are learning events currently stored/streamed?
3. **Item Catalog:** Where is item metadata stored? (API, database, etc.)
4. **Infrastructure:** What's the preferred deployment platform? (AWS, GCP, on-prem)
5. **Integration:** How should recommendations be delivered? (API, queue, webhook)
6. **SLA:** What are the latency/availability requirements?

