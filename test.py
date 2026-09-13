# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from backend import run_travel_agent

print("Starting test...")
user_input = "Plan a trip from New York to Paris including flights and hotels"
print(f"User input: {user_input}")
travel_agent_results = run_travel_agent(user_input)
print("Results received:")
print(travel_agent_results)
