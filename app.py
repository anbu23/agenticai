import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from typing import Optional
from api.datamodels import ApprovalRequest, TravelPlan
from db import db_utils

app = FastAPI(title="TravelMate AI API", version="1.0.0")

# ── Orchestrator Map ──────────────────────────────────────
ORCHESTRATOR_MAP = {}

def _get_orchestrator(phase: str):
    """Lazy-load orchestrators to avoid heavy imports at startup."""
    if phase not in ORCHESTRATOR_MAP:
        if phase == "phase2_crewai":
            from phases.phase2_crewai.trip_orchestrator import CrewAITripOrchestrator
            ORCHESTRATOR_MAP[phase] = CrewAITripOrchestrator()
        # Add phase3, phase4 here later
    return ORCHESTRATOR_MAP.get(phase)


# ── Endpoints ─────────────────────────────────────────────
@app.post("/api/v1/plan_trip")
def plan_trip(user_input: str, user_id: int, phase: str = "phase2_crewai"):
    """Plan a trip using the specified AI framework."""
    orch = _get_orchestrator(phase)
    if not orch:
        return {"success": False, "error": f"Unsupported phase: {phase}"}

    try:
        result = orch.plan_trip(
            user_input=user_input,
            user_id=user_id,
            trip_title="My Trip",
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v1/approve")
def approve_trip(request: ApprovalRequest):
    """Approve or reject a travel plan."""
    try:
        # Determine phase from trip
        trip = db_utils.get_trip_by_id(request.trip_id)
        if not trip:
            return {"success": False, "error": f"Trip {request.trip_id} not found"}

        phase = trip.phase
        orch = _get_orchestrator(phase)
        if not orch:
            return {"success": False, "error": f"No orchestrator for phase: {phase}"}

        decision = "approved" if request.approval else "rejected"
        result = orch.continue_trip_approval(
            trip_id=request.trip_id,
            user_id=request.user_id,
            approval_decision=decision,
            user_feedback=request.feedback or "",
        )

        result["trip_id"] = request.trip_id
        result["user_id"] = request.user_id
        result["approval"] = request.approval
        result["feedback"] = request.feedback

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/")
@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "service": "TravelMate AI API"}


@app.get("/api/v1/trip/{trip_id}/plan")
def get_trip_plan(trip_id: int, version: Optional[int] = None):
    try:
        tp = db_utils.get_trip_plan_by_trip_id(trip_id, version)
        if tp:
            return {
                "success": True,
                "plan": tp.to_travel_plan().model_dump(mode="json"),
                "metadata": {
                    "trip_id": tp.trip_id, "version": tp.version,
                    "status": tp.status, "generated_at": tp.generated_at,
                },
            }
        return {"success": False, "error": "Trip plan not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/v1/trip/{trip_id}/plan")
def save_trip_plan(trip_id: int, travel_plan: TravelPlan, version: int = 1):
    try:
        plan_id = db_utils.save_travel_plan_to_db(travel_plan, trip_id, version)
        return {"success": True, "plan_id": plan_id,
                "message": f"Trip plan saved for trip {trip_id}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.put("/api/v1/trip-plan/{plan_id}/status")
def update_plan_status(plan_id: int, status: str):
    try:
        updated = db_utils.update_trip_plan_status(plan_id, status)
        if updated:
            return {"success": True, "message": f"Plan status updated to {status}"}
        return {"success": False, "error": "Plan not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("Run: uvicorn api.app:app --reload")
