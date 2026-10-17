from agents import trace, gen_trace_id
from search_agent import perform_searches
from planner_agent import plan_searches
from writer_agent import write_report
from email_agent import send_email
from query_agent import refine_query
import asyncio

class ResearchManager:

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            yield f"Starting research. Trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            clarifying_questions = await refine_query(query)
            yield f"Clarifying questions: {clarifying_questions}"
            search_plan = await plan_searches(clarifying_questions)
            yield f"Searches planned, starting {len(search_plan.searches)} searches..."     
            search_results = await perform_searches(search_plan)
            yield "Searches complete, writing report..."
            report = await write_report(query, search_results)
            yield "Report written, sending email..."
            await send_email(report)
            yield "Email sent, research complete"
            yield report.markdown_report
