from agents import Agent, WebSearchTool, ModelSettings, Runner
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel, Field
load_dotenv(override=True)
MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")

NUMBER_OF_CLARIFYING_QUESTIONS = 3

INSTRUCTIONS = f"You are a helpful research assistant. Given a query, refine it with exactly {NUMBER_OF_CLARIFYING_QUESTIONS} \
clarifying questions so that the web search could be improved and the searches are more specific and focused. Return the \
list of clarifying questions and the refined query as your final output."

class ClarifyingQuestions(BaseModel):
    clarifying_questions: list[str] = Field(description="The clarifying questions to refine the query")
    refined_query: str = Field(description="The refined query")

query_agent = Agent[any](name="Query Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=ClarifyingQuestions)

async def refine_query(query: str) -> ClarifyingQuestions:
    """ Refine the query for clarifying questions """
    result = await Runner.run(query_agent, query)
    return result.final_output.refined_query