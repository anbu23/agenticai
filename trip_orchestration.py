"""
Phase 2: CrewAI Orchestrator — Sequential Agent Workflow
InfoCollector → Planner → Optimizer with DB persistence
"""

# Disable telemetry FIRST
import os
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import warnings
warnings.filterwarnings("ignore")

import time
import json
import re
import traceback
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

# Custom JSON encoder
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)

# CrewAI
from crewai import Crew, Task, Process

# Local imports
import db.db_utils as db_utils
from db.db_utils import (
    create_trip,
    update_trip_status,
    save_chat_message_service,
    save_travel_plan_to_db,
    get_trip_plan_by_trip_id,
    update_trip_plan_status,
)
from api.datamodels import (
    Trip, TripRequirements, TravelPlan, OptimizationResult,
    ChatHistory, HotelSuggestion, FlightSuggestion, TripPlanModel,
)
from phases.phase2_crewai.trip_agents import (
    info_collector, planner, optimizer,
)

PHASE = "phase2_crewai"


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM response (handles code fences)."""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    for pat in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```', r'(\{[\s\S]*\})']:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, cls=DateTimeEncoder, indent=2, default=str)
    except Exception:
        return str(obj)


def _log_chat(user_id: int, trip_id: Optional[int], role: str, content: str):
    """Save a chat message to DB (non-critical — never raises)."""
    try:
        msg = ChatHistory(
            user_id=user_id,
            trip_id=trip_id,
            role=role,
            phase=PHASE,
            content=content[:4000],
        )
        save_chat_message_service(msg)
    except Exception as e:
        print(f"⚠️ Chat log failed: {e}")


# ═══════════════════════════════════════════════════════════
#  TASK PROMPTS
# ═══════════════════════════════════════════════════════════
INFO_TASK_TMPL = """
Analyze the user's trip request and extract structured requirements.

USER INPUT: {user_input}
{history_block}

INSTRUCTIONS:
1. Use get_current_date to know today's date.
2. Resolve relative dates ("next Friday", "in 2 weeks") to YYYY-MM-DD.
3. Use search_web to validate destination info if needed.
4. If destination OR dates are missing, set mode="missing".

RESPOND WITH **ONLY** THIS JSON — no other text:
{{
  "mode": "trip",
  "origin": "departure city or 'Not specified'",
  "destination": "city name",
  "trip_startdate": "YYYY-MM-DD",
  "trip_enddate": "YYYY-MM-DD",
  "no_of_adults": 1,
  "no_of_children": 0,
  "budget": 1000.0,
  "currency": "USD",
  "accommodation_type": "hotel",
  "purpose": "leisure",
  "travel_preferences": "comma-separated preferences",
  "travel_constraints": "any constraints or 'none'",
  "missing_fields": [],
  "agent_message": ""
}}

If info is missing, use:
{{
  "mode": "missing",
  "error": "MISSING",
  "missing_fields": ["field1", "field2"],
  "agent_message": "Please provide: ..."
}}
"""

PLAN_TASK_TMPL = """
Create a complete travel plan using REAL data from the tools.

REQUIREMENTS:
{requirements_json}

INSTRUCTIONS:
1. Search flights from origin to destination (and return if round-trip).
2. Search hotels at the destination for the travel dates.
3. Check weather forecast for the destination.
4. Search experiences/activities at the destination.
5. If ANY tool fails or returns empty, IMMEDIATELY use search_web as fallback.
6. Create a day-by-day itinerary.

RESPOND WITH **ONLY** THIS JSON:
{{
  "itinerary": "Day 1: ...\\nDay 2: ...\\n...",
  "hotels": [
    {{
      "name": "Hotel Name",
      "price_per_night": 100.0,
      "rating": 4.5,
      "location": "City Center",
      "amenities": ["wifi", "breakfast"]
    }}
  ],
  "flights": [
    {{
      "airline": "Airline Name",
      "departure_time": "YYYY-MM-DD HH:MM",
      "arrival_time": "YYYY-MM-DD HH:MM",
      "price": 350.0,
      "duration": "5h 30m",
      "stops": 0
    }}
  ],
  "daily_budget": 150.0,
  "total_estimated_cost": 2500.0
}}
"""

OPT_TASK_TMPL = """
Optimize this travel plan for cost, timing, and customer satisfaction.

REQUIREMENTS:
{requirements_json}

CURRENT PLAN:
{plan_json}

INSTRUCTIONS:
1. Use search_web to find cheaper alternatives, deals, discount codes.
2. Rearrange itinerary for feasibility (clustering nearby activities, weather).
3. Suggest specific cost savings with dollar amounts.
4. List value-adds (free activities, happy hours, walking tours).

RESPOND WITH **ONLY** THIS JSON:
{{
  "recommendations": [
    "Specific recommendation 1 with estimated saving",
    "Specific recommendation 2"
  ],
  "cost_savings": 200.0,
  "value_adds": [
    "Free walking tour available on Day 2",
    "Happy hour at hotel bar 5-7pm"
  ],
  "final_plan": "Optimized day-by-day summary text",
  "approval_required": true
}}
"""


# ═══════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════
class CrewAITripOrchestrator:
    """
    Sequential CrewAI orchestrator: InfoCollector → Planner → Optimizer.
    All outputs persisted to DB via db_utils.
    """

    def __init__(self):
        self.info_agent = info_collector()
        self.planner_agent = planner()
        self.optimizer_agent = optimizer()

    # ──────────────────────────────────────────────────────
    #  MAIN ENTRY
    # ──────────────────────────────────────────────────────
    def plan_trip(
        self,
        user_input: str,
        user_id: int,
        trip_title: str = "My Trip",
        approval_callback=None,
        conversation_history: Optional[List[dict]] = None,
    ) -> dict:
        """
        Plan a trip end-to-end.
        Returns dict with: success, status, trip_id, message, requirements, plan, optimization
        """
        result = {
            "success": False,
            "status": "error",
            "trip_id": None,
            "message": "",
            "requirements": None,
            "plan": None,
            "optimization": None,
        }
        trip_id = None
        t0 = time.time()

        # Build conversation context
        history_block = ""
        if conversation_history:
            lines = [f"{m.get('role','user').upper()}: {m.get('content','')}"
                     for m in conversation_history[-10:]]
            history_block = "PREVIOUS CONVERSATION:\n" + "\n".join(lines)

        # Log user message
        _log_chat(user_id, None, "user", user_input)

        # ═════════════════════════════════════════════════
        # PHASE 1: INFO COLLECTION
        # ═════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("🔍 PHASE 1: Collecting trip requirements…")
        print("=" * 60)

        try:
            info_task = Task(
                description=INFO_TASK_TMPL.format(
                    user_input=user_input,
                    history_block=history_block,
                ),
                expected_output="JSON object matching TripRequirements schema",
                agent=self.info_agent,
            )
            crew = Crew(agents=[self.info_agent], tasks=[info_task],
                        process=Process.sequential, verbose=True)
            raw = str(crew.kickoff())
            print(f"📋 InfoCollector output: {raw[:500]}")

            req_dict = _extract_json(raw)
            if not req_dict:
                result["message"] = (
                    "I couldn't understand your request. Please provide:\n"
                    "• Destination\n• Travel dates\n• Number of travelers\n• Budget"
                )
                _log_chat(user_id, None, "assistant", result["message"])
                return result

            # Validate via Pydantic
            try:
                trip_req = TripRequirements(**req_dict)
            except ValidationError as ve:
                trip_req = TripRequirements(
                    mode="missing",
                    error="MISSING",
                    missing_fields=[str(e) for e in ve.errors()],
                    agent_message=f"Validation issues: {ve}",
                )

            result["requirements"] = trip_req.model_dump(mode="json")

            # Check if info is incomplete
            if not trip_req.is_complete():
                result["status"] = "need_info"
                result["message"] = trip_req.agent_message or trip_req.get_missing_info()
                _log_chat(user_id, None, "assistant", result["message"])
                return result

            print(f"✅ Requirements: {trip_req.origin} → {trip_req.destination}, "
                  f"{trip_req.trip_startdate} to {trip_req.trip_enddate}")

        except Exception as e:
            traceback.print_exc()
            result["message"] = f"Error collecting requirements: {e}"
            return result

        # ═════════════════════════════════════════════════
        # CREATE TRIP IN DB
        # ═════════════════════════════════════════════════
        try:
            trip_obj = Trip(
                user_id=user_id,
                phase=PHASE,
                title=trip_title or f"Trip to {trip_req.destination}",
                origin=trip_req.origin or "Not specified",
                destination=trip_req.destination or "Unknown",
                trip_startdate=trip_req.trip_startdate,
                trip_enddate=trip_req.trip_enddate,
                accommodation_type=trip_req.accommodation_type or "hotel",
                no_of_adults=trip_req.no_of_adults or 1,
                no_of_children=trip_req.no_of_children or 0,
                budget=trip_req.budget or 500.0,
                currency=trip_req.currency or "USD",
                trip_status="draft",
                purpose=trip_req.purpose or "leisure",
                travel_preferences=trip_req.travel_preferences or "none",
                travel_constraints=trip_req.travel_constraints or "none",
            )
            trip_id = create_trip(trip_obj)
            result["trip_id"] = trip_id
            print(f"💾 Trip created in DB: ID={trip_id}")

            update_trip_status(trip_id, "in_progress")
            _log_chat(user_id, trip_id, "system", f"Trip {trip_id} created. Starting planning.")

        except Exception as e:
            print(f"⚠️ Trip creation failed: {e}")
            traceback.print_exc()

        # ═════════════════════════════════════════════════
        # PHASE 2: PLANNING
        # ═════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("📅 PHASE 2: Creating travel plan…")
        print("=" * 60)

        travel_plan = None
        try:
            plan_task = Task(
                description=PLAN_TASK_TMPL.format(
                    requirements_json=_safe_json(result["requirements"]),
                ),
                expected_output="JSON matching TravelPlan schema with hotels, flights, itinerary",
                agent=self.planner_agent,
            )
            crew = Crew(agents=[self.planner_agent], tasks=[plan_task],
                        process=Process.sequential, verbose=True)
            raw = str(crew.kickoff())
            print(f"📅 Planner output: {raw[:500]}")

            plan_dict = _extract_json(raw)
            if plan_dict:
                try:
                    travel_plan = TravelPlan(**plan_dict)
                except ValidationError:
                    travel_plan = TravelPlan(
                        itinerary=plan_dict.get("itinerary", raw[:2000]),
                        hotels=plan_dict.get("hotels", []),
                        flights=plan_dict.get("flights", []),
                        daily_budget=float(plan_dict.get("daily_budget", 0)),
                        total_estimated_cost=float(plan_dict.get("total_estimated_cost", 0)),
                    )
            else:
                travel_plan = TravelPlan(itinerary=raw[:2000])

            result["plan"] = travel_plan.model_dump(mode="json")
            print(f"✅ Plan: {travel_plan.hotel_count()} hotels, "
                  f"{travel_plan.flight_count()} flights, "
                  f"est. ${travel_plan.total_estimated_cost or 0}")

            # Save plan to DB
            if trip_id:
                try:
                    plan_id = save_travel_plan_to_db(travel_plan, trip_id, version=1)
                    print(f"💾 Plan saved: plan_id={plan_id}")
                    _log_chat(user_id, trip_id, "assistant",
                              f"Travel plan created. Est. cost: ${travel_plan.total_estimated_cost or 'TBD'}")
                except Exception as e:
                    print(f"⚠️ Plan save failed: {e}")

        except Exception as e:
            traceback.print_exc()
            travel_plan = TravelPlan(itinerary=f"Planning error: {e}")
            result["plan"] = travel_plan.model_dump(mode="json")

        # ═════════════════════════════════════════════════
        # PHASE 3: OPTIMIZATION
        # ═════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("💰 PHASE 3: Optimizing…")
        print("=" * 60)

        optimization = None
        try:
            opt_task = Task(
                description=OPT_TASK_TMPL.format(
                    requirements_json=_safe_json(result["requirements"]),
                    plan_json=_safe_json(result["plan"]),
                ),
                expected_output="JSON matching OptimizationResult with recommendations and cost_savings",
                agent=self.optimizer_agent,
            )
            crew = Crew(agents=[self.optimizer_agent], tasks=[opt_task],
                        process=Process.sequential, verbose=True)
            raw = str(crew.kickoff())
            print(f"💰 Optimizer output: {raw[:500]}")

            opt_dict = _extract_json(raw)
            if opt_dict:
                try:
                    optimization = OptimizationResult(**opt_dict)
                except ValidationError:
                    optimization = OptimizationResult(
                        recommendations=opt_dict.get("recommendations", []),
                        cost_savings=float(opt_dict.get("cost_savings", 0)),
                        value_adds=opt_dict.get("value_adds", []),
                        final_plan=opt_dict.get("final_plan", raw[:1000]),
                        approval_required=True,
                    )
            else:
                optimization = OptimizationResult(final_plan=raw[:1000])

            result["optimization"] = optimization.model_dump(mode="json")
            print(f"✅ Optimization: ${optimization.cost_savings} savings, "
                  f"{len(optimization.recommendations)} recommendations")

            if trip_id:
                _log_chat(user_id, trip_id, "assistant",
                          f"Optimization complete. Savings: ${optimization.cost_savings}")

        except Exception as e:
            traceback.print_exc()
            optimization = OptimizationResult(final_plan=f"Optimization error: {e}")
            result["optimization"] = optimization.model_dump(mode="json")

        # ═════════════════════════════════════════════════
        # FINALIZE
        # ═════════════════════════════════════════════════
        elapsed = round(time.time() - t0, 1)
        print(f"\n⏱️ Total: {elapsed}s")

        if trip_id:
            update_trip_status(trip_id, "draft")

        result["success"] = True
        result["status"] = "pending_approval"
        result["message"] = self._build_summary(trip_req, travel_plan, optimization, elapsed)

        _log_chat(user_id, trip_id, "assistant", result["message"])

        # Approval callback
        if approval_callback and trip_id:
            try:
                decision = approval_callback(result)
                if decision:
                    return self.continue_trip_approval(
                        trip_id, user_id,
                        "approved" if decision.get("approved") else "rejected",
                        decision.get("feedback", ""),
                    )
            except Exception:
                pass

        return result

    # ──────────────────────────────────────────────────────
    #  APPROVAL
    # ──────────────────────────────────────────────────────
    def continue_trip_approval(
        self,
        trip_id: int,
        user_id: int,
        approval_decision: str,
        user_feedback: str = "",
    ) -> dict:
        result = {
            "success": True,
            "trip_id": trip_id,
            "status": "error",
            "message": "",
            "updated_status": "",
            "plan_id": None,
        }

        # Get latest plan
        plan = get_trip_plan_by_trip_id(trip_id)
        if plan:
            result["plan_id"] = plan.id

        if approval_decision.lower() in ("approved", "approve", "yes", "true"):
            update_trip_status(trip_id, "confirmed")
            if plan:
                update_trip_plan_status(plan.id, "approved")
            result["status"] = "complete"
            result["updated_status"] = "approved"
            result["message"] = "✅ Travel plan approved successfully! Have an amazing trip! 🌍✈️"
            _log_chat(user_id, trip_id, "system", "Plan approved by user.")

        elif approval_decision.lower() in ("rejected", "reject", "no", "false"):
            update_trip_status(trip_id, "draft")
            if plan:
                update_trip_plan_status(plan.id, "rejected")
            result["updated_status"] = "rejected"
            if user_feedback:
                result["status"] = "rejected"
                result["message"] = (
                    f"Travel plan rejected. Feedback: \"{user_feedback}\"\n"
                    "You can start a new plan with updated preferences."
                )
                _log_chat(user_id, trip_id, "user", f"Rejected: {user_feedback}")
            else:
                result["status"] = "rejected"
                result["message"] = "Plan rejected. Would you like to plan again with different preferences?"
            _log_chat(user_id, trip_id, "system", "Plan rejected by user.")
        else:
            result["message"] = f"Unknown decision '{approval_decision}'. Use 'approved' or 'rejected'."

        return result

    # ──────────────────────────────────────────────────────
    #  SUMMARY BUILDER
    # ──────────────────────────────────────────────────────
    def _build_summary(self, req: TripRequirements, plan: TravelPlan,
                       opt: OptimizationResult, elapsed: float) -> str:
        dest = req.destination or "your destination"
        start = str(req.trip_startdate or "TBD")
        end = str(req.trip_enddate or "TBD")

        # Flights
        flight_lines = []
        for f in plan.flights[:2]:
            flight_lines.append(
                f"  ✈️ {f.airline} — ${f.price:.0f} | {f.duration} | {f.stops_display()}"
            )

        # Hotels
        hotel_lines = []
        for h in plan.hotels[:2]:
            hotel_lines.append(
                f"  🏨 {h.name} — {h.price_display()} | {h.rating_display()}"
            )

        # Optimizations
        opt_lines = [f"  💡 {r}" for r in opt.recommendations[:3]]

        summary = f"""
🌍 **Trip to {dest}** ({start} → {end})
👥 {req.no_of_adults} adult(s), {req.no_of_children} child(ren) | 💰 {req.budget} {req.currency}
{'─' * 40}

{''.join(chr(10) + l for l in flight_lines) if flight_lines else '  ✈️ Flight details in full plan'}
{''.join(chr(10) + l for l in hotel_lines) if hotel_lines else '  🏨 Hotel details in full plan'}

📋 **Itinerary**: {plan.itinerary_text()[:300]}...

💰 **Estimated Cost**: ${plan.total_estimated_cost or 'TBD'}
💡 **Potential Savings**: ${opt.cost_savings}
{''.join(chr(10) + l for l in opt_lines) if opt_lines else ''}

⏱️ Planned in {elapsed}s

**Would you like to approve this plan or request changes?**
        """.strip()

        return summary


# ═══════════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════════
def test_orchestrator():
    orch = CrewAITripOrchestrator()
    result = orch.plan_trip(
        user_input=(
            "I want to plan a leisure trip from Bangalore to Goa "
            "from 2025-08-10 to 2025-08-14, for 2 adults "
            "with a budget of 8000 INR."
        ),
        user_id=1,
        trip_title="Goa Beach Trip",
    )
    print("\n" + "=" * 60)
    print("📊 RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Status:  {result['status']}")
    print(f"Trip ID: {result['trip_id']}")
    print(f"\n{result['message']}")
    with open("test_trip_result.json", "w") as f:
        json.dump(result, f, cls=DateTimeEncoder, indent=2, default=str)


if __name__ == "__main__":
    test_orchestrator()
