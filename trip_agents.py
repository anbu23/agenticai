"""
Phase 2: CrewAI Agents — Tool wrappers + Agent factories
"""

from crewai import Agent
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional, Type
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
#  LLM HELPER
# ═══════════════════════════════════════════════════════════
def _get_llm():
    """Return the shared ChatOpenAI instance for all agents."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )


# ═══════════════════════════════════════════════════════════
#  TOOL INPUT SCHEMAS (Pydantic)
# ═══════════════════════════════════════════════════════════
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query string")


class WeatherInput(BaseModel):
    city: str = Field(description="City name (e.g. 'Paris')")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")


class WeatherSimpleInput(BaseModel):
    city: str = Field(description="City name")
    days: int = Field(default=3, description="Number of forecast days (1-16)")


class HotelSearchInput(BaseModel):
    city: str = Field(description="City name")
    check_in: str = Field(description="Check-in date YYYY-MM-DD")
    check_out: str = Field(description="Check-out date YYYY-MM-DD")
    adults: int = Field(default=1, description="Number of adults")


class FlightSearchInput(BaseModel):
    origin: str = Field(description="Origin city name")
    destination: str = Field(description="Destination city name")
    departure_date: str = Field(description="Departure date YYYY-MM-DD")
    return_date: Optional[str] = Field(default=None, description="Return date YYYY-MM-DD (optional)")
    adults: int = Field(default=1, description="Number of adults")


class ExperienceSearchInput(BaseModel):
    city: str = Field(description="City name")
    radius_km: int = Field(default=20, description="Search radius in km")
    max_results: int = Field(default=10, description="Max results")


class DateTimeInput(BaseModel):
    timezone: Optional[str] = Field(default=None, description="Timezone e.g. 'UTC'")


# ═══════════════════════════════════════════════════════════
#  CREWAI TOOL WRAPPERS
# ═══════════════════════════════════════════════════════════
class SearchWebTool(BaseTool):
    name: str = "search_web"
    description: str = (
        "Search the web for travel info, visa requirements, local tips, "
        "safety advisories, cultural events, etc. "
        "Input: a search query string."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            service = WebSearchService()
            result = service.search(query, max_results=5)
            if "error" in result:
                return f"Search error: {result['error']}"
            # Compact results for LLM context
            summaries = []
            for r in result.get("results", []):
                summaries.append(
                    f"• {r['title']}\n  {r['content'][:300]}\n  URL: {r['url']}"
                )
            return f"Web search results for '{query}':\n\n" + "\n\n".join(summaries)
        except Exception as e:
            return f"Web search failed: {str(e)}"


class GetWeatherTool(BaseTool):
    name: str = "get_weather_range"
    description: str = (
        "Get weather forecast for a city over a date range. "
        "Provide city name, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)."
    )
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str, start_date: str, end_date: str) -> str:
        try:
            tool = WeatherTool()
            result = tool.get_weather_range(city, start_date, end_date)
            if "error" in result:
                # Fall back to simple forecast
                simple = tool.get_weather(city, days=7)
                if "error" in simple:
                    return f"Weather error: {simple['error']}"
                result = simple
            loc = result.get("location", {})
            header = f"Weather for {loc.get('name', city)}, {loc.get('country', '')}:\n"
            lines = []
            for day in result.get("forecast", []):
                lines.append(
                    f"  {day['date']}: {day['temp_min']}°C – {day['temp_max']}°C, "
                    f"precip: {day['precipitation']}mm"
                )
            return header + "\n".join(lines)
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"


class SearchHotelsTool(BaseTool):
    name: str = "search_hotels"
    description: str = (
        "Search for hotels in a city with check-in/out dates. "
        "Returns hotel names, prices, ratings, and amenities. "
        "Provide city, check_in (YYYY-MM-DD), check_out (YYYY-MM-DD), adults."
    )
    args_schema: Type[BaseModel] = HotelSearchInput

    def _run(self, city: str, check_in: str, check_out: str, adults: int = 1) -> str:
        try:
            toolkit = AmadeusHotelToolkit()
            hotel_ids, hotels = toolkit.hotel_list(city, radius=10)
            if not hotel_ids:
                return f"No hotels found in {city}."

            # Limit to top 8 to avoid API limits
            hotel_ids = hotel_ids[:8]
            hotels = hotels[:8]

            offers = toolkit.hotel_search(hotel_ids, hotels, check_in, check_out, adults)

            if not offers:
                # Return basic hotel list without offers
                lines = [f"Hotels found in {city} (no live pricing available):"]
                for h in hotels:
                    lines.append(
                        f"• {h.get('name', 'Unknown')} "
                        f"(ID: {h.get('hotelId', 'N/A')}, "
                        f"Dist: {h.get('distance', {}).get('value', '?')} "
                        f"{h.get('distance', {}).get('unit', '')})"
                    )
                return "\n".join(lines)

            # Format offers compactly
            lines = [f"Hotel offers in {city} ({check_in} to {check_out}):"]
            for offer_data in offers[:6]:
                hotel_info = offer_data.get("hotel", {})
                name = hotel_info.get("name", "Unknown Hotel")
                for off in offer_data.get("offers", []):
                    price = off.get("price", {})
                    room = off.get("room", {})
                    room_desc = room.get("description", {}).get("text", "N/A")
                    lines.append(
                        f"• {name}: ${price.get('total', '?')} {price.get('currency', 'USD')} "
                        f"| Room: {room_desc[:80]} "
                        f"| Check-in: {off.get('checkInDate', check_in)} "
                        f"| Check-out: {off.get('checkOutDate', check_out)}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Hotel search failed: {str(e)}"


class SearchFlightsTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search for flights between two cities. "
        "Returns airlines, prices, duration, and stops. "
        "Provide origin, destination, departure_date (YYYY-MM-DD), "
        "optional return_date, adults."
    )
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(self, origin: str, destination: str, departure_date: str,
             return_date: Optional[str] = None, adults: int = 1) -> str:
        try:
            toolkit = AmadeusFlightToolkit()
            offers = toolkit.flight_search(
                origin, destination, departure_date,
                return_date=return_date, adults=adults
            )
            if not offers:
                return f"No flights found from {origin} to {destination} on {departure_date}."

            lines = [f"Flights from {origin} to {destination} on {departure_date}:"]
            for offer in offers[:4]:
                price = offer.get("price", {})
                airlines = ", ".join(offer.get("validatingAirlineCodes", []))
                for itin in offer.get("itineraries", []):
                    duration = itin.get("duration", "N/A")
                    segments = itin.get("segments", [])
                    stops = len(segments) - 1
                    dep = segments[0].get("departure", {}) if segments else {}
                    arr = segments[-1].get("arrival", {}) if segments else {}
                    lines.append(
                        f"• ${price.get('total', '?')} {price.get('currency', 'USD')} "
                        f"| {airlines} | {duration} | {stops} stop(s) "
                        f"| {dep.get('iataCode', '?')} {dep.get('at', '?')[:16]} "
                        f"→ {arr.get('iataCode', '?')} {arr.get('at', '?')[:16]}"
                    )
            return "\n".join(lines)
        except Exception as e:
            return f"Flight search failed: {str(e)}"


class SearchExperiencesTool(BaseTool):
    name: str = "search_experiences"
    description: str = (
        "Search for tours, activities, and experiences in a city. "
        "Returns activity names, prices, ratings, and descriptions. "
        "Provide city name, optional radius_km and max_results."
    )
    args_schema: Type[BaseModel] = ExperienceSearchInput

    def _run(self, city: str, radius_km: int = 20, max_results: int = 10) -> str:
        try:
            toolkit = AmadeusExperienceToolkit()
            activities = toolkit.experience_search(city, radius_km=radius_km, max_results=max_results)
            if not activities:
                return f"No experiences/activities found in {city}."

            lines = [f"Experiences & activities in {city}:"]
            for act in activities:
                price = act.get("price", {})
                lines.append(
                    f"• {act.get('name', 'Unknown')} "
                    f"| ${price.get('amount', '?')} {price.get('currencyCode', 'USD')} "
                    f"| Rating: {act.get('rating', 'N/A')} "
                    f"| {(act.get('shortDescription', '') or '')[:120]}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Experience search failed: {str(e)}"


class GetCurrentDateTool(BaseTool):
    name: str = "get_current_date"
    description: str = (
        "Get today's current date and time. Use this to determine "
        "relative dates like 'next week' or 'in 3 days'. "
        "Optional timezone parameter."
    )
    args_schema: Type[BaseModel] = DateTimeInput

    def _run(self, timezone: Optional[str] = None) -> str:
        try:
            dt_tool = DateTimeTool()
            result = dt_tool.get_current_datetime(timezone=timezone)
            if "error" in result:
                return f"DateTime error: {result['error']}"
            return (
                f"Current date: {result['date']}\n"
                f"Current time: {result['time']}\n"
                f"Timezone: {result['timezone']}"
            )
        except Exception as e:
            return f"DateTime failed: {str(e)}"


# ═══════════════════════════════════════════════════════════
#  AGENT FACTORIES
# ═══════════════════════════════════════════════════════════
def info_collector() -> Agent:
    """
    Agent that extracts structured trip requirements from user input.
    Uses web search + current date to validate and enrich info.
    Returns TripRequirements as JSON.
    """
    return Agent(
        role="Trip Requirements Analyst",
        goal=(
            "Extract complete, structured trip requirements from the user's "
            "natural language request. Identify destination, origin, dates, "
            "budget, number of travelers, preferences (hotel star rating, "
            "activities, dietary needs), and any special requirements. "
            "If critical info is missing (destination or dates), clearly "
            "list what's needed. Use today's date to resolve relative dates "
            "like 'next weekend' or 'in 2 weeks'."
        ),
        backstory=(
            "You are an expert travel consultant who has helped thousands "
            "of clients clarify their travel needs. You're meticulous about "
            "details — you never assume, you always confirm. You know that "
            "a great trip starts with great requirements gathering."
        ),
        tools=[
            SearchWebTool(),
            GetCurrentDateTool(),
        ],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def planner() -> Agent:
    """
    Agent that creates a detailed travel plan using real API data.
    Searches flights, hotels, weather, and experiences.
    Returns TravelPlan as JSON.
    """
    return Agent(
        role="Travel Planner",
        goal=(
            "Create a comprehensive, day-by-day travel plan with REAL data. "
            "Search for actual flights with prices, real hotel options with "
            "rates, check the weather forecast, and find local experiences. "
            "Produce a complete itinerary with specific recommendations, "
            "booking references, and a cost breakdown."
        ),
        backstory=(
            "You are a world-class travel planner with 20 years of experience "
            "crafting unforgettable trips. You have access to live flight, "
            "hotel, and activity databases. You always provide multiple "
            "options at different price points and consider weather, travel "
            "time between locations, and local events. You create practical, "
            "realistic itineraries — not fantasy trips."
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
    """
    Agent that optimizes a travel plan for cost, value, and logistics.
    Returns OptimizationResult as JSON.
    """
    return Agent(
        role="Travel Plan Optimizer",
        goal=(
            "Review and optimize the travel plan for best value. "
            "Identify cost savings (cheaper flights at nearby times, "
            "better-value hotels, bundle deals). Check for logistics "
            "issues (tight connections, long transfers, overbooking). "
            "Suggest alternatives that save money without sacrificing "
            "experience quality. Provide a final optimized plan with "
            "total cost comparison (original vs optimized)."
        ),
        backstory=(
            "You are a travel optimization expert and former airline "
            "revenue analyst. You know every trick to get the best "
            "value: flexible date savings, loyalty programs, off-peak "
            "timing, package deals, and hidden-city ticketing. You "
            "balance cost savings with traveler comfort and experience "
            "quality. You always explain WHY each optimization helps."
        ),
        tools=[
            SearchFlightsTool(),
            SearchHotelsTool(),
            SearchWebTool(),
        ],
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )
