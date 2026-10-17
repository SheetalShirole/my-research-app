from agents import Agent, WebSearchTool, ModelSettings, Runner
from dotenv import load_dotenv
from WebSearchModel import WebSearchItem, WebSearchPlan
import os
import asyncio
load_dotenv(override=True)
MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")

INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and 
produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""
settings= ModelSettings(model=MODEL_NAME,tool_choice="required")
tool=[WebSearchTool()]
search_agent = Agent[any](name="Search Agent", instructions=INSTRUCTIONS, model_settings=settings, tools=tool)

async def perform_searches(search_plan: WebSearchPlan) -> list[str]:
    """ Perform the searches to perform for the query """
    tasks = [search(item) for item in search_plan.searches]
    return await asyncio.gather(*tasks)

async def search(item: WebSearchItem) -> str | None:
    """ Perform a search for the query """
    input_message = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, input_message)
    return result.final_output