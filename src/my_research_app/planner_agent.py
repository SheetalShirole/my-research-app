from pydantic import BaseModel, Field
from agents import Agent, Runner
from WebSearchModel import WebSearchPlan
import os
from dotenv import load_dotenv
load_dotenv(override=True)

MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")
HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", 5))


INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""


planner_agent = Agent[any](name="Planner Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=WebSearchPlan)

async def plan_searches(query: str) -> WebSearchPlan:
    """ Plan the searches to perform for the query """
    result = await Runner.run(planner_agent, f"Query: {query}")
    return result.final_output

