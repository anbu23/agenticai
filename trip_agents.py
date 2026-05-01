"""
Phase 2: CrewAI Agents — Tool wrappers + Agent factories
Aligned with datamodels.py (TripRequirements, TravelPlan, OptimizationResult)
"""

from crewai import Agent
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Optional, Type
from pydantic import BaseModel, Field
import json
import os

# Import tool services
from toolkits.web_search_service import WebSearchService
from toolkits.weather_tool import WeatherTool
from toolkits.amadeus_hotel_search import AmadeusHotelToolkit
from toolkits.amadeus_flight_tool import AmadeusFlightToolkit
from toolkits.amadeus_experience_tool import AmadeusExperienceToolkit
from toolkits.current_datetime import DateTimeTool
from api.datamodels import TripRequirements, TravelPlan, OptimizationResult


# ═══════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════
def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )


# ═══════════════════════════════════════════════════════════
#  TOOL INPUT SCHEMAS
# ═══════════════════════════════════════════════════════════
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")

class WeatherRangeInput(BaseModel):
    city: str = Field(description="City name")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")

class HotelInput(BaseModel):
    city: str = Field(description="City name")
    check_in: str = Field(description="Check-in date YYYY-MM-DD")
    check_out: str = Field(description="Check-out date YYYY-MM-DD")
    adults: int = Field(default=1, description="Number of adults")

class FlightInput(BaseModel):
    origin: str = Field(description="Origin city name")
    destination: str = Field(description="Destination city name")
    departure_date: str = Field(description="Departure date YYYY-MM-DD")
    return_date: Optional[str] = Field(default=None, description="Return date YYYY-MM-DD")
    adults: int = Field(default=1, description="Number of adults")

class ExperienceInput(BaseModel):
    city: str = Field(description="City name")
    radius_km: int = Field(default=20, description="Search radius km")
    max_results: int = Field(default=10, description="Max results")

class DateTimeInput(BaseModel):
    timezone: Optional[str] = Field(default=None, description="Timezone e.g. 'UTC'")


# ═══════════════════════════════════════════════════════════
#  CREWAI TOOL WRAPPERS
# ═══════════════════════════════════════════════════════════
class SearchWebTool(BaseTool):
    """🔄 FALLBACK TOOL — use whenever any other tool fails or returns empty."""
    name: str = "search_web"
    description: str = (
        "Search the web for travel info, visa requirements, local tips, "
        "safety advisories, deals, prices, alternatives. "
        "USE THIS AS FALLBACK when any other tool fails or returns no results. "
        "Input: a search query string."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            svc = WebSearchService()
            result = svc.search(query, max_results=5)
            if "error" in result:
                return f"Search error: {result['error']}"
            lines = []
            for r in result.get("results", []):
                lines.append(f"• {r['title']}\n  {r['content'][:250]}\n  {r['url']}")
            return f"Results for '{query}':\n\n" + "\n\n".join(lines) if lines else "No results found."
        except Exception as e:
            return f"Web search failed: {e}"


class GetWeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = (
        "Get weather forecast for a city and date range. "
        "Provide city, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD). "
        "If this fails, use search_web as fallback."
    )
    args_schema: Type[BaseModel] = WeatherRangeInput

    def _run(self, city: str, start_date: str, end_date: str) -> str:
        try:
            wt = WeatherTool()
            result = wt.get_weather_range(city, start_date, end_date)
            if "error" in result:
                simple = wt.get_weather(city, days=7)
                if "error" in simple:
                    return f"Weather unavailable: {simple['error']}. Use search_web as fallback."
                result = simple
            loc = result.get("location", {})
            lines = [f"Weather for {loc.get('name', city)}, {loc.get('country', '')}:"]
            for d in result.get("forecast", []):
                lines.append(f"  {d['date']}: {d['temp_min']}°C–{d['temp_max']}°C, rain: {d['precipitation']}mm")
            return "\n".join(lines)
        except Exception as e:
            return f"Weather failed: {e}. Use search_web as fallback."


class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = (
        "Search for hotels in a city with check-in/out dates using Amadeus API. "
        "Returns hotel names, prices, amenities. "
        "If this fails or returns empty, use search_web as fallback."
    )
    args_schema: Type[BaseModel] = HotelInput

    def _run(self, city: str, check_in: str, check_out: str, adults: int = 1) -> str:
        try:
            tk = AmadeusHotelToolkit()
            ids, hotels = tk.hotel_list(city, radius=10)
            if not ids:
                return f"No hotels found in {city} via Amadeus. Use search_web as fallback."
            ids, hotels = ids[:8], hotels[:8]
            offers = tk.hotel_search(ids, hotels, check_in, check_out, adults)
            if not offers:
                lines = [f"Hotels in {city} (no live pricing):"]
                for h in hotels[:5]:
                    lines.append(f"• {h.get('name','?')} — {h.get('distance',{}).get('value','?')} {h.get('distance',{}).get('unit','')}")
                return "\n".join(lines)
            lines = [f"Hotels in {city} ({check_in} to {check_out}):"]
            for od in offers[:6]:
                nm = od.get("hotel", {}).get("name", "Unknown")
                for off in od.get("offers", []):
                    p = off.get("price", {})
                    rm = off.get("room", {}).get("description", {}).get("text", "N/A")
                    lines.append(f"• {nm}: ${p.get('total','?')} {p.get('currency','USD')} | {rm[:80]}")
            return "\n".join(lines)
        except Exception as e:
            return f"Hotel search failed: {e}. Use search_web as fallback."


class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search for flights between two cities using Amadeus API. "
        "Returns airlines, prices, duration, stops. "
        "If this fails or returns empty, use search_web as fallback."
    )
    args_schema: Type[BaseModel] = FlightInput

    def _run(self, origin: str, destination: str, departure_date: str,
             return_date: Optional[str] = None, adults: int = 1) -> str:
        try:
            tk = AmadeusFlightToolkit()
            offers = tk.flight_search(origin, destination, departure_date,
                                      return_date=return_date, adults=adults)
            if not offers:
                return f"No flights {origin}→{destination} on {departure_date}. Use search_web as fallback."
            lines = [f"Flights {origin}→{destination} on {departure_date}:"]
            for offer in offers[:4]:
                price = offer.get("price", {})
                airlines = ", ".join(offer.get("validatingAirlineCodes", []))
                for itin in offer.get("itineraries", []):
                    segs = itin.get("segments", [])
                    dep = segs[0].get("departure", {}) if segs else {}
                    arr = segs[-1].get("arrival", {}) if segs else {}
                    lines.append(
                        f"• ${price.get('total','?')} {price.get('currency','USD')} | {airlines} "
                        f"| {itin.get('duration','?')} | {len(segs)-1} stop(s) "
                        f"| {dep.get('iataCode','?')} {dep.get('at','?')[:16]} → "
                        f"{arr.get('iataCode','?')} {arr.get('at','?')[:16]}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Flight search failed: {e}. Use search_web as fallback."


class SearchExperiencesTool(BaseTool):
    name: str = "search_experiences"
    description: str = (
        "Search for tours, activities in a city using Amadeus API. "
        "If this fails, use search_web as fallback."
    )
    args_schema: Type[BaseModel] = ExperienceInput

    def _run(self, city: str, radius_km: int = 20, max_results: int = 10) -> str:
        try:
            tk = AmadeusExperienceToolkit()
            acts = tk.experience_search(city, radius_km=radius_km, max_results=max_results)
            if not acts:
                return f"No experiences in {city} via Amadeus. Use search_web as fallback."
            lines = [f"Experiences in {city}:"]
            for a in acts:
                p = a.get("price", {})
                lines.append(
                    f"• {a.get('name','?')} | ${p.get('amount','?')} {p.get('currencyCode','USD')} "
                    f"| Rating: {a.get('rating','N/A')} | {(a.get('shortDescription','') or '')[:100]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Experience search failed: {e}. Use search_web as fallback."


class GetCurrentDateTool(BaseTool):
    name: str = "get_current_date"
    description: str = "Get today's date. Use to resolve relative dates like 'next week'."
    args_schema: Type[BaseModel] = DateTimeInput

    def _run(self, timezone: Optional[str] = None) -> str:
        try:
            dt = DateTimeTool()
            result = dt.get_current_datetime(timezone=timezone)
            if "error" in result:
                return f"Error: {result['error']}"
            return f"Today's date: {result['date']} | Time: {result['time']} | TZ: {result['timezone']}"
        except Exception as e:
            return f"DateTime failed: {e}"


# ═══════════════════════════════════════════════════════════
#  AGENT FACTORIES
# ═══════════════════════════════════════════════════════════
def info_collector() -> Agent:
    """InfoCollector — extracts TripRequirements from user input."""
    return Agent(
        role="Travel Requirements Specialist",
        goal=(
            "Extract and validate complete travel requirements from user requests. "
            "Resolve relative dates using get_current_date. "
            "If critical info is missing, set mode='missing' with missing_fields list. "
            "Output must be a valid TripRequirements JSON."
        ),
        backstory=(
            "You are an experienced travel consultant who specializes in "
            "understanding customer needs and gathering comprehensive travel "
            "requirements. You always validate destinations and dates."
        ),
        tools=[SearchWebTool(), GetCurrentDateTool()],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def planner() -> Agent:
    """Planner — creates TravelPlan with real API data."""
    return Agent(
        role="Travel Itinerary Specialist",
        goal=(
            "Create comprehensive travel itineraries with REAL flights, hotels, "
            "and activities from the tools. Always try the specialized tools first. "
            "If ANY tool fails or returns no results, IMMEDIATELY use search_web "
            "as fallback. Never return empty sections. "
            "Output must match TravelPlan JSON format."
        ),
        backstory=(
            "You are a skilled travel planner with extensive knowledge of "
            "destinations worldwide and access to the best travel booking systems. "
            "You always provide practical, realistic itineraries with real pricing."
        ),
        tools=[
            SearchFlightsTool(),
            SearchHotelsTool(),
            SearchExperiencesTool(),
            GetWeatherTool(),
            SearchWebTool(),
            GetCurrentDateTool(),
        ],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def optimizer() -> Agent:
    """Optimizer — optimizes plan for cost/value, outputs OptimizationResult."""
    return Agent(
        role="Travel Cost Optimizer",
        goal=(
            "Optimize travel plans for cost, timing, and satisfaction. "
            "Use search_web extensively to find cheaper alternatives and deals. "
            "Provide specific recommendations with estimated savings. "
            "Rearrange itinerary for feasibility, clustering, and weather. "
            "Output must match OptimizationResult JSON format."
        ),
        backstory=(
            "You are a financial analyst specializing in travel cost optimization "
            "and finding the best value propositions for customers. You know every "
            "trick to save money without sacrificing experience quality."
        ),
        tools=[SearchWebTool()],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )
