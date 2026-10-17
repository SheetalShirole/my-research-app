from agents import trace, gen_trace_id,function_tool, Agent,Runner, ModelSettings
from search_agent import perform_searches
from planner_agent import plan_searches
from writer_agent import write_report
from email_agent import send_email
from query_agent import refine_query
import asyncio
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class ResearchManager:
    MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")

    task = """
    You are a research assistant. You will be given a query and you will need to refine it with clarifying questions.
    You will then need to plan the searches to perform for the query. You will then need to perform the searches and
    write a report based on the search results. You will then need to send the report via email.
    You will need to use the tools provided to you to perform the tasks.
    """
    settings = ModelSettings(tool_choice="required")
    refine_query_tool = function_tool(refine_query)
    plan_searches_tool = function_tool(plan_searches)
    perform_searches_tool = function_tool(perform_searches)
    write_report_tool = function_tool(write_report)
    send_email_tool = function_tool(send_email)
    research_manager_agent = Agent(name="Research Manager", instructions=task, tools=[refine_query_tool, plan_searches_tool, perform_searches_tool, write_report_tool, send_email_tool], model=MODEL_NAME, model_settings=settings)

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            yield f"Starting research. Trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            result = await Runner.run(self.research_manager_agent, query)
            yield result.final_output


    # async def run(self, query: str):
    #     """ Run the deep research process, yielding the status updates and the final report"""
    #     trace_id = gen_trace_id()
    #     with trace("Research trace", trace_id=trace_id):
    #         yield f"Starting research. Trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
    #         clarifying_questions = await refine_query(query)
    #         yield f"Clarifying questions: {clarifying_questions}"
    #         search_plan = await plan_searches(clarifying_questions)
    #         yield f"Searches planned, starting {len(search_plan.searches)} searches..."     
    #         search_results = await perform_searches(search_plan)
    #         yield "Searches complete, writing report..."
    #         report = await write_report(query, search_results)
    #         yield "Report written, sending email..."
    #         await send_email(report)
    #         yield "Email sent, research complete"
    #         yield report.markdown_report
