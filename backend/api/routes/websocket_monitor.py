from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.orm import Session
from typing import Dict, List
import asyncio
import json
import numpy as np
from datetime import datetime, timezone, timedelta
import random
from api.models.database import get_db, Model, BiasAnalysis, ExternalModel, PredictionLog
from core.src.logger import logging
from core.auth.jwt import decode_access_token
from api.models.user import User

router = APIRouter()


async def verify_websocket_token(token: str, db: Session) -> User:
    """Verify JWT token for WebSocket connections"""
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise ValueError("Invalid token")
        
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user or not user.is_active:
            raise ValueError("User not found or inactive")
        
        return user
    except Exception as e:
        logging.error(f"WebSocket auth failed: {e}")
        raise ValueError("Authentication failed")

def check_model_access(user: User, model_org_id: str) -> bool:
    """Check if user can access model"""
    # Superuser can access everything
    if user.is_superuser:
        return True
    
    # User must be in same organization
    return user.organization_id == model_org_id


class ConnectionManager:
    """Manage WebSocket connections for real-time monitoring"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logging.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.append(connection)
        
        for conn in dead_connections:
            self.disconnect(conn)

manager = ConnectionManager()


class EnhancedBiasMonitor:
    """
    Enhanced bias monitoring for BiasGuard 2.0
    Works with both trained models and external models
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.alert_thresholds = {
            "disparate_impact_min": 0.8,
            "disparate_impact_max": 1.25,
            "statistical_parity": 0.1,
            "equal_opportunity": 0.1,
            "drift_warning": 0.1,
            "drift_critical": 0.15
        }
    
    def calculate_drift_score(self, model_id: str) -> Dict:
        """Calculate model drift from baseline with velocity"""
        
        recent_analyses = self.db.query(BiasAnalysis).filter(
            BiasAnalysis.model_id == model_id
        ).order_by(BiasAnalysis.analyzed_at.desc()).limit(10).all()
        
        if len(recent_analyses) < 2:
            return {
                "drift_score": 0,
                "status": "stable",
                "trend": "stable",
                "velocity": 0
            }
        
        # Parse metrics
        if isinstance(recent_analyses[-1].fairness_metrics, dict):
            baseline_metrics = recent_analyses[-1].fairness_metrics
        else:
            baseline_metrics = json.loads(recent_analyses[-1].fairness_metrics or '{}')
        
        if isinstance(recent_analyses[0].fairness_metrics, dict):
            current_metrics = recent_analyses[0].fairness_metrics
        else:
            current_metrics = json.loads(recent_analyses[0].fairness_metrics or '{}')
        
        # Calculate drift
        drift_score = 0
        drift_count = 0
        
        for attr in baseline_metrics.keys():
            if attr in current_metrics:
                if 'disparate_impact' in baseline_metrics[attr] and 'disparate_impact' in current_metrics[attr]:
                    base_di = baseline_metrics[attr]['disparate_impact'].get('ratio', 1.0)
                    curr_di = current_metrics[attr]['disparate_impact'].get('ratio', 1.0)
                    
                    if isinstance(base_di, (int, float)) and isinstance(curr_di, (int, float)):
                        if base_di not in [float('inf'), float('-inf')] and curr_di not in [float('inf'), float('-inf')]:
                            drift_score += abs(curr_di - base_di)
                            drift_count += 1
                
                if 'statistical_parity' in baseline_metrics[attr] and 'statistical_parity' in current_metrics[attr]:
                    base_sp = baseline_metrics[attr]['statistical_parity'].get('statistical_parity_diff', 0)
                    curr_sp = current_metrics[attr]['statistical_parity'].get('statistical_parity_diff', 0)
                    drift_score += abs(curr_sp - base_sp)
                    drift_count += 1
        
        drift_score = drift_score / max(1, drift_count)
        drift_score = max(0, drift_score + random.uniform(-0.01, 0.01))
        
        velocity = drift_score / max(1, len(recent_analyses) - 1)
        
        if drift_score > self.alert_thresholds["drift_critical"]:
            status = "critical"
        elif drift_score > self.alert_thresholds["drift_warning"]:
            status = "warning"
        else:
            status = "stable"
        
        return {
            "drift_score": round(drift_score, 4),
            "status": status,
            "trend": "increasing" if drift_score > 0.1 else "stable",
            "velocity": round(velocity, 4)
        }
    
    def _calculate_trends(self, analyses: List[BiasAnalysis]) -> Dict:
        """Calculate fairness metric trends over time"""
        
        if len(analyses) < 2:
            return {}
        
        trends = {}
        metrics_history = {
            "disparate_impact": [],
            "statistical_parity": [],
            "equal_opportunity": []
        }
        
        for analysis in reversed(analyses[:10]):
            fm = analysis.fairness_metrics if isinstance(analysis.fairness_metrics, dict) else json.loads(analysis.fairness_metrics or '{}')
            
            first_attr = list(fm.keys())[0] if fm else None
            if not first_attr:
                continue
            
            attr_m = fm[first_attr]
            
            if 'disparate_impact' in attr_m:
                di_val = attr_m['disparate_impact'].get('ratio', 1.0)
                if isinstance(di_val, (int, float)) and di_val not in [float('inf'), float('-inf')]:
                    metrics_history["disparate_impact"].append(di_val)
            
            if 'statistical_parity' in attr_m:
                sp_val = abs(attr_m['statistical_parity'].get('statistical_parity_diff', 0))
                metrics_history["statistical_parity"].append(sp_val)
            
            if 'equal_opportunity' in attr_m:
                eo_val = abs(attr_m['equal_opportunity'].get('difference', 0))
                metrics_history["equal_opportunity"].append(eo_val)
        
        # Analyze trends
        for metric_name, values in metrics_history.items():
            if len(values) >= 2:
                recent = values[-3:] if len(values) >= 3 else values
                older = values[:3] if len(values) >= 3 else values
                
                recent_avg = float(np.mean(recent))
                older_avg = float(np.mean(older))
                change = recent_avg - older_avg
                
                if metric_name == "disparate_impact":
                    direction = "improving" if abs(recent_avg - 1.0) < abs(older_avg - 1.0) else ("degrading" if abs(recent_avg - 1.0) > abs(older_avg - 1.0) else "stable")
                else:
                    direction = "improving" if change < -0.01 else ("degrading" if change > 0.01 else "stable")
                
                trends[metric_name] = {
                    "direction": direction,
                    "change_rate": float(change),
                    "recent_avg": recent_avg,
                    "baseline_avg": older_avg
                }
        
        return trends
    
    def _generate_alerts(self, fairness_metrics: Dict, drift_data: Dict) -> List[Dict]:
        """Generate prioritized alerts with regulatory context"""
        
        alerts = []
        
        for attr, metrics in fairness_metrics.items():
            if 'disparate_impact' in metrics:
                di = metrics['disparate_impact']
                ratio = di.get('ratio', 1.0)
                
                if isinstance(ratio, (int, float)) and ratio not in [float('inf'), float('-inf')]:
                    if ratio < 0.8 or ratio > 1.25:
                        alerts.append({
                            "type": "critical",
                            "metric": "disparate_impact",
                            "attribute": attr,
                            "value": ratio,
                            "threshold": "0.8-1.25",
                            "message": f"Disparate Impact violation for {attr}: {ratio:.3f}",
                            "regulation": "ECOA"
                        })
                    elif ratio < 0.85 or ratio > 1.20:
                        alerts.append({
                            "type": "warning",
                            "metric": "disparate_impact",
                            "attribute": attr,
                            "value": ratio,
                            "threshold": "0.8-1.25",
                            "message": f"Disparate Impact approaching threshold for {attr}: {ratio:.3f}"
                        })
            
            if 'statistical_parity' in metrics:
                sp = metrics['statistical_parity']
                diff = abs(sp.get('statistical_parity_diff', 0))
                
                if diff > 0.1:
                    alerts.append({
                        "type": "critical",
                        "metric": "statistical_parity",
                        "attribute": attr,
                        "value": diff,
                        "threshold": 0.1,
                        "message": f"Statistical parity violation for {attr}: {diff:.3f}",
                        "regulation": "Title VII"
                    })
        
        if drift_data["status"] == "critical":
            alerts.append({
                "type": "critical",
                "metric": "model_drift",
                "value": drift_data["drift_score"],
                "threshold": 0.15,
                "message": f"Critical model drift: {drift_data['drift_score']:.4f}"
            })
        
        return alerts
    
    async def get_realtime_metrics(self, model_id: str) -> Dict:
        """
        Get comprehensive real-time metrics
        Works with BOTH ExternalModel and Model
        """
        
        # Try ExternalModel first
        external_model = self.db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id
        ).first()
        
        if external_model:
            # BiasGuard 2.0 model
            model_name = external_model.model_name
            
            # Get recent predictions from logs
            cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_logs = self.db.query(PredictionLog).filter(
                PredictionLog.model_id == model_id,
                PredictionLog.logged_at >= cutoff
            ).all()
            
            predictions_last_hour = len(recent_logs)
            
            # Calculate average confidence if available
            confidences = [log.prediction_proba for log in recent_logs if log.prediction_proba]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.75
            
        else:
            # BiasGuard 1.0 model (legacy)
            model = self.db.query(Model).filter(Model.model_id == model_id).first()
            if not model:
                return {"error": "Model not found"}
            
            model_name = model.model_type
            predictions_last_hour = random.randint(1000, 5000)  # Simulated
            avg_confidence = round(random.uniform(0.75, 0.95), 3)
        
        # Get latest analysis (works for both)
        analyses = self.db.query(BiasAnalysis).filter(
            BiasAnalysis.model_id == model_id
        ).order_by(BiasAnalysis.analyzed_at.desc()).limit(10).all()
        
        if not analyses:
            return {
                "model_id": model_id,
                "status": "no_analysis",
                "message": "No bias analysis available"
            }
        
        latest_analysis = analyses[0]
        
        # Calculate drift
        drift_data = self.calculate_drift_score(model_id)
        
        # Build response
        current_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "model_name": model_name,
            "predictions_last_hour": predictions_last_hour,
            "avg_confidence": avg_confidence,
            "drift_status": drift_data,
            "current_bias_metrics": {},
            "alerts": [],
            "trends": {},
            "performance": {}
        }
        
        # Parse metrics
        metrics = latest_analysis.fairness_metrics if isinstance(latest_analysis.fairness_metrics, dict) else json.loads(latest_analysis.fairness_metrics or '{}')
        
        # Extract current bias metrics
        first_attr = list(metrics.keys())[0] if metrics else None
        if first_attr and first_attr in metrics:
            attr_m = metrics[first_attr]
            
            di_val = attr_m.get("disparate_impact", {}).get("ratio", 1.0)
            if isinstance(di_val, str):
                di_val = 1.0  # Handle "inf" string
            
            current_metrics["current_bias_metrics"] = {
                "disparate_impact": di_val,
                "statistical_parity": attr_m.get("statistical_parity", {}).get("statistical_parity_diff", 0),
                "equal_opportunity": attr_m.get("equal_opportunity", {}).get("difference", 0),
                "accuracy": 0.75  # Default for external models
            }
        
        # Calculate trends
        current_metrics["trends"] = self._calculate_trends(analyses)
        
        # Performance tracking (simulated for now)
        current_metrics["performance"] = {
            "baseline_accuracy": 0.75,
            "current_accuracy": 0.75 + random.uniform(-0.02, 0.02),
            "accuracy_drop": abs(random.uniform(0, 0.03))
        }
        
        # Generate alerts
        current_metrics["alerts"] = self._generate_alerts(metrics, drift_data)
        
        return current_metrics



# WEBSOCKET ENDPOINTS
@router.websocket("/monitor/{model_id}")
async def monitor_model(websocket: WebSocket, model_id: str, token: str = Query(...)):
    """
    Real-time monitoring for ANY model (BiasGuard 1.0 or 2.0)
    
    Streams every 5 seconds:
    - Fairness metrics
    - Drift analysis
    - Trends
    - Alerts
    """
    
    
    db = next(get_db())
    monitor = EnhancedBiasMonitor(db)
    
    try:
        
        logging.info(f"Starting monitoring for {model_id}")
        user = await verify_websocket_token(token, db)
        logging.info(f"WebSocket auth: {user.username} ({user.role})")
        
        #model and check access
        model = db.query(ExternalModel).filter(
            ExternalModel.model_id == model_id
        ).first()
        
        if not model:
            # Try legacy Model table
            model = db.query(Model).filter(Model.model_id == model_id).first()
        
        if not model:
            await websocket.close(code=1008, reason="Model not found")
            return
        
        #Check organization access
        if not check_model_access(user, model.organization_id):
            logging.warning(f"Access denied: {user.username} to {model_id}")
            await websocket.close(code=1008, reason="Access denied - different organization")
            return
        
        # Accept connection after auth
        await manager.connect(websocket)
        logging.info(f"Monitoring {model_id} for {user.username}")
        
        monitor = EnhancedBiasMonitor(db)
        
        # Send initial data
        initial_data = await monitor.get_realtime_metrics(model_id)
        await websocket.send_json(initial_data)
        # Send initial data
        initial_data = await monitor.get_realtime_metrics(model_id)
        await websocket.send_json(initial_data)
        
        # Continuous monitoring loop
        update_count = 0
        while True:
            await asyncio.sleep(5)
            update_count += 1
            
            metrics = await monitor.get_realtime_metrics(model_id)
            
            # Add real-time simulation
            if "current_bias_metrics" in metrics and metrics["current_bias_metrics"]:
                for key in metrics["current_bias_metrics"]:
                    if key != "accuracy" and isinstance(metrics["current_bias_metrics"][key], (int, float)):
                        current_val = metrics["current_bias_metrics"][key]
                        variation = random.uniform(-0.005, 0.005)
                        metrics["current_bias_metrics"][key] = round(current_val + variation, 4)
            
            await websocket.send_json(metrics)
            
            if update_count % 12 == 0:
                logging.info(f"{model_id}: {len(metrics.get('alerts', []))} alerts")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        db.close()
        logging.info(f"Client disconnected from {model_id}")
    except Exception as e:
        logging.error(f"Monitor error: {str(e)}")
        db.close()


@router.websocket("/dashboard")
async def dashboard_monitor(websocket: WebSocket, token: str = Query(...)):
    """
    Dashboard-level monitoring (all models)
    
    Streams every 3 seconds:
    - System health
    - Models at risk
    - Platform statistics
    """
    
    
    db = next(get_db())
    
    try:
        
        logging.info("Starting dashboard monitoring")
        user = await verify_websocket_token(token, db)
        logging.info(f"Dashboard WebSocket: {user.username} ({user.role})")
        
        await manager.connect(websocket)
        logging.info(f"Dashboard monitoring for {user.username}")
        
        while True:
            # Get all models (both types)
            trained_models = db.query(Model).all()
            external_models = db.query(ExternalModel).all()
            
            dashboard_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_models": len(trained_models) + len(external_models),
                "models_at_risk": 0,
                "system_health": {
                    "api_latency_ms": random.randint(10, 50),
                    "active_connections": len(manager.active_connections),
                    "predictions_per_second": random.randint(50, 200)
                },
                "models_status": []
            }
            
            monitor = EnhancedBiasMonitor(db)
            
            # Check trained models
            for model in trained_models[:5]:
                try:
                    drift_data = monitor.calculate_drift_score(model.model_id)
                    status = "healthy"
                    
                    if model.bias_status == "critical":
                        status = "critical"
                        dashboard_data["models_at_risk"] += 1
                    elif model.bias_status == "warning" or drift_data["status"] in ["warning", "critical"]:
                        status = "warning"
                        dashboard_data["models_at_risk"] += 1
                    
                    dashboard_data["models_status"].append({
                        "model_id": model.model_id,
                        "model_type": model.model_type,
                        "source": "trained",
                        "status": status,
                        "drift_score": drift_data["drift_score"],
                        "last_updated": model.updated_at.isoformat()
                    })
                except Exception as e:
                    logging.error(f"Error checking trained model {model.model_id}: {e}")
            
            # Check external models
            for ext_model in external_models[:5]:
                try:
                    drift_data = monitor.calculate_drift_score(ext_model.model_id)
                    
                    # Get latest analysis
                    latest_analysis = db.query(BiasAnalysis).filter(
                        BiasAnalysis.model_id == ext_model.model_id
                    ).order_by(BiasAnalysis.analyzed_at.desc()).first()
                    
                    status = "healthy"
                    if latest_analysis:
                        if latest_analysis.bias_status == "critical":
                            status = "critical"
                            dashboard_data["models_at_risk"] += 1
                        elif latest_analysis.bias_status == "warning":
                            status = "warning"
                            dashboard_data["models_at_risk"] += 1
                    
                    dashboard_data["models_status"].append({
                        "model_id": ext_model.model_id,
                        "model_type": ext_model.model_name,
                        "source": "external",
                        "status": status,
                        "drift_score": drift_data["drift_score"],
                        "last_updated": ext_model.updated_at.isoformat()
                    })
                except Exception as e:
                    logging.error(f"Error checking external model {ext_model.model_id}: {e}")
            
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(3)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        db.close()
        logging.info("Dashboard client disconnected")
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        db.close()



# REST ENDPOINTS

@router.get("/monitoring/alerts/{model_id}")
async def get_model_alerts(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current alerts for a model (REST endpoint)
    Works with both BiasGuard 1.0 and 2.0 models
    """
    
    monitor = EnhancedBiasMonitor(db)
    metrics = await monitor.get_realtime_metrics(model_id)
    
    return {
        "model_id": model_id,
        "alerts": metrics.get("alerts", []),
        "drift_status": metrics.get("drift_status", {}),
        "trends": metrics.get("trends", {}),
        "performance": metrics.get("performance", {}),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }