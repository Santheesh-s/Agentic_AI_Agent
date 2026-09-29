"""The Info Agent, implemented as a ReAct agent with an iteration limit.

The course rubric requires "at least one ReAct agent with an iteration
limit". This module builds one with LangGraph's prebuilt create_react_agent:
given a query, the agent decides for itself which tools to call (the three
MCP-derived tools plus the two extra LangChain tools) and in what order,
observes each tool's result, and repeats until it has enough information to
answer or it hits the iteration cap - which is enforced by passing
recursion_limit in the invoke config (see run_info_agent below). This
replaces a fixed "always call these steps" pipeline with genuine agentic
tool selection.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from assistant.extra_tools import otc_dosage_limits, web_search_tool
from mcp_server.tools import check_interaction, drug_info, find_generic

# Each ReAct "step" consumes ~2 recursion units (one for the model call, one for
# the tool call), so a cap of 10 allows up to ~5 tool calls before the agent
# must answer with what it has - the iteration limit required by the rubric.
MAX_ITERATIONS = 5
RECURSION_LIMIT = MAX_ITERATIONS * 2

SYSTEM_PROMPT = (
    "You are the Info Agent of a medicine information assistant. Use the available tools "
    "to gather facts before answering: drug_info for a medicine's uses/side effects/warnings, "
    "find_generic for brand-to-generic lookups, check_interaction for interaction checks, "
    "otc_dosage_limits for factual OTC label maximums, and web_search_tool for current "
    "advisories or recalls. Never invent facts a tool did not return. Never state a "
    "personalised dose, diagnose a condition, or tell the user to start/stop a medicine - "
    "state label facts only. Keep the final answer concise and in plain language."
)


def build_info_agent(llm):
    tools = [drug_info, find_generic, check_interaction, otc_dosage_limits, web_search_tool]
    return create_react_agent(llm, tools)


from assistant.llm import extract_text_content


def run_info_agent(llm, query: str) -> str:
    """Runs the ReAct Info Agent on one query, capped at RECURSION_LIMIT steps."""
    agent = build_info_agent(llm)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    try:
        result = agent.invoke({"messages": messages}, config={"recursion_limit": RECURSION_LIMIT})
    except Exception as exc:
        # GraphRecursionError (iteration limit hit) or a tool/model failure: degrade, don't crash.
        return f"[Info Agent stopped after reaching its iteration limit or an error ({exc.__class__.__name__})]"
    final = result["messages"][-1]
    return extract_text_content(final.content)

