# Disable telemetry FIRST
import os
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

import warnings
warnings.filterwarnings("ignore")

import time
import json
import traceback
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from pydantic import ValidationError

# Custom JSON encoder for date objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)

# CrewAI imports
from crewai import Crew, Task, Process

# Local imports
import db.db_utils as db_utils
from api.datamodels import (
    TripRequirements, Trip, TravelPlan,
    OptimizationResult, ChatHistory,
)
from db.db_utils import save_chat_message
from phases.phase2_crewai.trip_agents import (
    info_collector, planner, optimizer,
)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def _extract_json(text: str) -> Optional[dict]:
    """
    Extract the first JSON object from an LLM response.
    Handles markdown code fences and surrounding text.
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fence
    import re
    patterns = [
        r'```json\s*([\s\S]*?)```',
        r'```\s*([\s\S]*?)```',
        r'(\{[\s\S]*\})',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    return None


def _safe_serialize(obj: Any) -> str:
    """Serialize any object to JSON string safely."""
    try:
        return json.dumps(obj, cls=DateTimeEncoder, indent=2, default=str)
    except Exception:
        return str(obj)


# ═══════════════════════════════════════════════════════════
#  TASK DESCRIPTIONS (templates)
# ═══════════════════════════════════════════════════════════
INFO_COLLECTOR_TASK = """
Analyze the user's trip request and extract structured requirements.

USER INPUT: {user_input}

{history_context}

INSTRUCTIONS:
1. Use the get_current_date tool to determine today's date.
2. Extract ALL trip details from the user input.
3. Resolve relative dates (e.g. "next Friday", "in 2 weeks") to actual YYYY-MM-DD dates.
4. If the destination or travel dates are completely missing, set "missing_info" with what's needed.
5. Search the web for any destination-specific info that helps (visa, safety, events).

RESPOND WITH ONLY THIS JSON (no other text):
{{
    "destination": "city/country name",
    "origin": "departure city (or 'Not specified')",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "budget": "total budget as number or 'Not specified'",
    "currency": "USD/EUR/etc",
    "travelers": {{
        "adults": 1,
        "children": 0
    }},
    "preferences": {{
        "hotel_stars": "3-5 or 'any'",
        "interests": ["list", "of", "interests"],
        "dietary": "any dietary requirements",
        "mobility": "any mobility requirements",
        "other": "any other preferences"
    }},
    "trip_type": "leisure/business/adventure/romantic/family",
    "missing_info": [],
    "destination_notes": "visa info, safety notes, relevant events"
}}
"""

PLANNER_TASK = """
Create a complete travel plan using REAL data from the tools.

TRIP REQUIREMENTS:
{requirements}

INSTRUCTIONS:
1. Search for flights from origin to destination (and return).
2. Search for hotels at the destination for the travel dates.
3. Check the weather forecast for the destination.
4. Search for experiences and activities at the destination.
5. Create a day-by-day itinerary.
6. Provide at least 2 options at different price points where possible.

RESPOND WITH ONLY THIS JSON:
{{
    "flights": {{
        "outbound": [
            {{
                "airline": "airline name",
                "flight_number": "XX123",
                "departure": "YYYY-MM-DD HH:MM",
                "arrival": "YYYY-MM-DD HH:MM",
                "origin": "airport code",
                "destination": "airport code",
                "price": 000.00,
                "currency": "USD",
                "duration": "XhYm",
                "stops": 0
            }}
        ],
        "return": []
    }},
    "hotels": [
        {{
            "name": "hotel name",
            "stars": 4,
            "price_per_night": 000.00,
            "total_price": 000.00,
            "currency": "USD",
            "location": "area/neighborhood",
            "amenities": ["list"],
            "check_in": "YYYY-MM-DD",
            "check_out": "YYYY-MM-DD"
        }}
    ],
    "weather": {{
        "summary": "overall weather summary",
        "daily": [
            {{"date": "YYYY-MM-DD", "high": 00, "low": 00, "conditions": "desc"}}
        ]
    }},
    "experiences": [
        {{
            "name": "activity name",
            "price": 00.00,
            "currency": "USD",
            "duration": "X hours",
            "description": "brief description",
            "rating": 4.5,
            "suggested_day": 1
        }}
    ],
    "daily_itinerary": [
        {{
            "day": 1,
            "date": "YYYY-MM-DD",
            "title": "Day title",
            "activities": ["morning: ...", "afternoon: ...", "evening: ..."]
        }}
    ],
    "cost_breakdown": {{
        "flights": 000.00,
        "hotels": 000.00,
        "experiences": 000.00,
        "meals_estimate": 000.00,
        "transport_estimate": 000.00,
        "total_estimate": 000.00,
        "currency": "USD"
    }},
    "travel_tips": ["tip1", "tip2"]
}}
"""

OPTIMIZER_TASK = """
Review and optimize this travel plan for best value.

TRIP REQUIREMENTS:
{requirements}

CURRENT TRAVEL PLAN:
{travel_plan}

INSTRUCTIONS:
1. Review the current plan for cost efficiency.
2. Search for potentially cheaper flight alternatives (nearby dates/times).
3. Check if there are better-value hotel options.
4. Identify any logistics issues (tight connections, long waits).
5. Suggest specific optimizations with estimated savings.
6. Produce the final optimized plan.

RESPOND WITH ONLY THIS JSON:
{{
    "optimizations": [
        {{
            "category": "flights/hotels/activities/logistics",
            "original": "what was planned",
            "suggested": "what you suggest instead",
            "savings": 00.00,
            "reason": "why this is better"
        }}
    ],
    "optimized_plan": {{
        "flights": {{ ... }},
        "hotels": [ ... ],
        "experiences": [ ... ],
        "daily_itinerary": [ ... ],
        "cost_breakdown": {{ ... }}
    }},
    "cost_comparison": {{
        "original_total": 000.00,
        "optimized_total": 000.00,
        "total_savings": 000.00,
        "currency": "USD"
    }},
    "risk_notes": ["any risks or caveats with optimizations"],
    "final_recommendation": "summary of the best plan and why"
}}
"""


# ═══════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════
class CrewAITripOrchestrator:
    """
    Orchestrator for Phase 2: CrewAI Sequential Agent Workflow.
    Runs InfoCollector → Planner → Optimizer in sequence.
    """

    def __init__(self):
        self.info_agent = info_collector()
        self.planner_agent = planner()
        self.optimizer_agent = optimizer()

    # ──────────────────────────────────────────────────────
    #  MAIN ENTRY POINT
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

        Returns:
            dict with keys:
              status: "complete" | "need_info" | "pending_approval" | "error"
              trip_id: int (if saved)
              requirements: dict
              travel_plan: dict
              optimization: dict
              message: str (user-facing summary)
        """
        result = {
            "status": "error",
            "trip_id": None,
            "requirements": None,
            "travel_plan": None,
            "optimization": None,
            "message": "",
        }

        start_time = time.time()

        # ── Build history context ──────────────────────────
        history_context = ""
        if conversation_history:
            lines = []
            for msg in conversation_history[-10:]:  # last 10 messages
                role = msg.get("role", "user")
                text = msg.get("content", "")
                lines.append(f"{role.upper()}: {text}")
            history_context = (
                "PREVIOUS CONVERSATION:\n" + "\n".join(lines) + "\n"
            )

        # ── Save user message to DB ────────────────────────
        try:
            save_chat_message(
                user_id=user_id,
                trip_id=None,
                role="user",
                content=user_input,
            )
        except Exception:
            pass  # non-critical

        # ═══ PHASE 1: INFO COLLECTION ═════════════════════
        print("\n" + "=" * 60)
        print("🔍 PHASE 1: Collecting trip requirements…")
        print("=" * 60)

        try:
            info_task = Task(
                description=INFO_COLLECTOR_TASK.format(
                    user_input=user_input,
                    history_context=history_context,
                ),
                expected_output="JSON object with structured trip requirements",
                agent=self.info_agent,
            )

            info_crew = Crew(
                agents=[self.info_agent],
                tasks=[info_task],
                process=Process.sequential,
                verbose=True,
            )

            info_result = info_crew.kickoff()
            info_text = str(info_result)
            print(f"\n📋 InfoCollector raw output:\n{info_text[:500]}…")

            requirements = _extract_json(info_text)
            if not requirements:
                result["status"] = "error"
                result["message"] = (
                    "I couldn't parse the trip requirements. "
                    "Could you rephrase your request with destination and dates?"
                )
                return result

            # Check for missing critical info
            missing = requirements.get("missing_info", [])
            if missing:
                result["status"] = "need_info"
                result["requirements"] = requirements
                result["message"] = (
                    "I need a bit more information to plan your trip:\n"
                    + "\n".join(f"• {item}" for item in missing)
                    + "\n\nPlease provide these details and I'll get started!"
                )
                self._save_assistant_message(user_id, None, result["message"])
                return result

            result["requirements"] = requirements
            print(f"✅ Requirements extracted: {requirements.get('destination')}, "
                  f"{requirements.get('start_date')} → {requirements.get('end_date')}")

        except Exception as e:
            traceback.print_exc()
            result["message"] = f"Error collecting requirements: {str(e)}"
            return result

        # ═══ PHASE 2: PLANNING ════════════════════════════
        print("\n" + "=" * 60)
        print("📅 PHASE 2: Creating travel plan…")
        print("=" * 60)

        try:
            plan_task = Task(
                description=PLANNER_TASK.format(
                    requirements=_safe_serialize(requirements),
                ),
                expected_output="JSON object with complete travel plan",
                agent=self.planner_agent,
            )

            plan_crew = Crew(
                agents=[self.planner_agent],
                tasks=[plan_task],
                process=Process.sequential,
                verbose=True,
            )

            plan_result = plan_crew.kickoff()
            plan_text = str(plan_result)
            print(f"\n📅 Planner raw output:\n{plan_text[:500]}…")

            travel_plan = _extract_json(plan_text)
            if not travel_plan:
                # Use raw text as plan summary
                travel_plan = {"raw_plan": plan_text, "cost_breakdown": {}}

            result["travel_plan"] = travel_plan
            total = travel_plan.get("cost_breakdown", {}).get("total_estimate", "N/A")
            print(f"✅ Travel plan created. Estimated cost: ${total}")

        except Exception as e:
            traceback.print_exc()
            result["travel_plan"] = {"error": str(e), "raw_plan": "Planning failed"}
            result["message"] = f"Planning partially failed: {str(e)}"

        # ═══ PHASE 3: OPTIMIZATION ════════════════════════
        print("\n" + "=" * 60)
        print("💰 PHASE 3: Optimizing travel plan…")
        print("=" * 60)

        try:
            opt_task = Task(
                description=OPTIMIZER_TASK.format(
                    requirements=_safe_serialize(requirements),
                    travel_plan=_safe_serialize(travel_plan),
                ),
                expected_output="JSON object with optimization results",
                agent=self.optimizer_agent,
            )

            opt_crew = Crew(
                agents=[self.optimizer_agent],
                tasks=[opt_task],
                process=Process.sequential,
                verbose=True,
            )

            opt_result = opt_crew.kickoff()
            opt_text = str(opt_result)
            print(f"\n💰 Optimizer raw output:\n{opt_text[:500]}…")

            optimization = _extract_json(opt_text)
            if not optimization:
                optimization = {"raw_optimization": opt_text}

            result["optimization"] = optimization
            savings = optimization.get("cost_comparison", {}).get("total_savings", 0)
            print(f"✅ Optimization complete. Potential savings: ${savings}")

        except Exception as e:
            traceback.print_exc()
            result["optimization"] = {"error": str(e)}

        # ═══ SAVE & FINALIZE ══════════════════════════════
        elapsed = round(time.time() - start_time, 1)
        print(f"\n⏱️ Total pipeline time: {elapsed}s")

        try:
            # Create Trip in DB
            trip_id = db_utils.create_trip(
                user_id=user_id,
                title=trip_title or f"Trip to {requirements.get('destination', 'Unknown')}",
                destination=requirements.get("destination", ""),
                start_date=requirements.get("start_date", ""),
                end_date=requirements.get("end_date", ""),
                requirements_json=_safe_serialize(requirements),
                plan_json=_safe_serialize(travel_plan),
                optimization_json=_safe_serialize(optimization),
                status="pending_approval",
            )
            result["trip_id"] = trip_id
            print(f"💾 Trip saved to DB with ID: {trip_id}")
        except Exception as e:
            print(f"⚠️ DB save failed (non-critical): {e}")
            # Try minimal save
            try:
                trip_id = db_utils.create_trip(
                    user_id=user_id,
                    title=trip_title,
                    destination=requirements.get("destination", "Unknown"),
                    start_date=requirements.get("start_date", ""),
                    end_date=requirements.get("end_date", ""),
                )
                result["trip_id"] = trip_id
            except Exception:
                pass

        # ── Build user-facing summary ──────────────────────
        result["status"] = "pending_approval"
        result["message"] = self._build_summary(requirements, travel_plan, optimization, elapsed)

        # Save assistant response
        self._save_assistant_message(user_id, result.get("trip_id"), result["message"])

        # ── Approval callback ──────────────────────────────
        if approval_callback and result["trip_id"]:
            try:
                decision = approval_callback(result)
                if decision:
                    return self.continue_trip_approval(
                        result["trip_id"],
                        decision.get("approved", "approved"),
                        decision.get("feedback", ""),
                    )
            except Exception:
                pass  # approval handled async

        return result

    # ──────────────────────────────────────────────────────
    #  APPROVAL CONTINUATION
    # ──────────────────────────────────────────────────────
    def continue_trip_approval(
        self,
        trip_id: int,
        approval_decision: str,
        user_feedback: str = "",
    ) -> dict:
        """
        Continue after user approves or rejects the plan.

        Args:
            trip_id: Trip database ID
            approval_decision: 'approved' or 'rejected'
            user_feedback: Optional feedback text

        Returns:
            dict with status and message
        """
        result = {
            "status": "error",
            "trip_id": trip_id,
            "message": "",
        }

        if approval_decision.lower() in ("approved", "approve", "yes"):
            # Mark trip as approved
            try:
                db_utils.update_trip_status(trip_id, "approved")
                result["status"] = "complete"
                result["message"] = (
                    "✅ Your trip has been approved and saved! "
                    "You can view it in your trip history. "
                    "Have an amazing journey! 🌍✈️"
                )
            except Exception as e:
                result["message"] = f"Approval saved but DB update failed: {e}"
                result["status"] = "complete"

        elif approval_decision.lower() in ("rejected", "reject", "no"):
            # Mark as rejected, optionally re-plan
            try:
                db_utils.update_trip_status(trip_id, "rejected")
            except Exception:
                pass

            if user_feedback:
                result["status"] = "replanning"
                result["message"] = (
                    f"Got it — I'll revise the plan based on your feedback: "
                    f"\"{user_feedback}\"\n\nRe-planning now…"
                )
                # Could trigger re-planning here with feedback
            else:
                result["status"] = "rejected"
                result["message"] = (
                    "The plan has been rejected. "
                    "Would you like to start over with different preferences?"
                )
        else:
            result["message"] = (
                f"Unknown decision: '{approval_decision}'. "
                "Please respond with 'approved' or 'rejected'."
            )

        return result

    # ──────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────
    def _build_summary(
        self,
        requirements: dict,
        travel_plan: dict,
        optimization: dict,
        elapsed: float,
    ) -> str:
        """Build a user-friendly summary of the trip plan."""
        dest = requirements.get("destination", "your destination")
        start = requirements.get("start_date", "TBD")
        end = requirements.get("end_date", "TBD")

        # Cost info
        cost = travel_plan.get("cost_breakdown", {})
        original_total = cost.get("total_estimate", "N/A")
        currency = cost.get("currency", "USD")

        opt_cost = optimization.get("cost_comparison", {})
        optimized_total = opt_cost.get("optimized_total", original_total)
        savings = opt_cost.get("total_savings", 0)

        # Flights summary
        flights = travel_plan.get("flights", {})
        outbound = flights.get("outbound", [])
        flight_info = ""
        if outbound:
            f = outbound[0]
            flight_info = (
                f"✈️ **Flight**: {f.get('airline', 'N/A')} "
                f"({f.get('origin', '?')} → {f.get('destination', '?')}) "
                f"— ${f.get('price', 'N/A')}"
            )

        # Hotel summary
        hotels = travel_plan.get("hotels", [])
        hotel_info = ""
        if hotels:
            h = hotels[0]
            hotel_info = (
                f"🏨 **Hotel**: {h.get('name', 'N/A')} "
                f"({'⭐' * h.get('stars', 0)}) "
                f"— ${h.get('total_price', h.get('price_per_night', 'N/A'))}"
            )

        # Experiences
        experiences = travel_plan.get("experiences", [])
        exp_count = len(experiences)

        # Optimizations
        opts = optimization.get("optimizations", [])
        opt_lines = []
        for o in opts[:3]:
            opt_lines.append(
                f"  💡 {o.get('category', '').title()}: {o.get('reason', '')}"
            )

        # Itinerary
        itinerary = travel_plan.get("daily_itinerary", [])
        itin_lines = []
        for day in itinerary[:5]:
            itin_lines.append(
                f"  📌 Day {day.get('day', '?')}: {day.get('title', 'TBD')}"
            )

        summary = f"""
🌍 **Trip to {dest}** ({start} → {end})
{'─' * 40}

{flight_info}
{hotel_info}
🎯 **Activities**: {exp_count} experience(s) found

📋 **Itinerary Preview**:
{chr(10).join(itin_lines) if itin_lines else '  (see full plan)'}

💰 **Cost Summary**:
  Original estimate: ${original_total} {currency}
  Optimized estimate: ${optimized_total} {currency}
  Potential savings: ${savings} {currency}

{"🔧 **Top Optimizations**:" if opt_lines else ""}
{chr(10).join(opt_lines)}

⏱️ Planned in {elapsed}s

**Would you like to approve this plan or make changes?**
        """.strip()

        return summary

    def _save_assistant_message(self, user_id: int, trip_id: Optional[int], content: str):
        """Save an assistant chat message to DB."""
        try:
            save_chat_message(
                user_id=user_id,
                trip_id=trip_id,
                role="assistant",
                content=content,
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════════
def test_orchestrator():
    """Test the orchestrator with a sample request."""
    orch = CrewAITripOrchestrator()

    test_input = (
        "I want to plan a 5-day trip to Paris next month. "
        "Budget around $3000 for 2 adults. "
        "We love art museums, good food, and walking tours. "
        "Flying from New York. Mid-range hotels preferred."
    )

    print("🚀 Starting test trip planning…")
    print(f"📝 Input: {test_input}\n")

    result = orch.plan_trip(
        user_input=test_input,
        user_id=1,
        trip_title="Paris Art & Food Trip",
    )

    print("\n" + "=" * 60)
    print("📊 FINAL RESULT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Trip ID: {result['trip_id']}")
    print(f"\n{result['message']}")

    # Save full result
    with open("test_trip_result.json", "w") as f:
        json.dump(result, f, cls=DateTimeEncoder, indent=2, default=str)
    print("\n💾 Full result saved to test_trip_result.json")


if __name__ == "__main__":
    test_orchestrator()
