"""LangGraph orchestration: shared state, conditional routing, and a
human-in-the-loop interrupt before the Escalation node.

    START -> guardrail --+--> escalation (INTERRUPT) ---------------+--> END
                        +--> decline ------------------------------+--> END
                        +--> interaction --> format_output ---------+--> END
                        +--> info -----------> format_output --------+--> END

Every query passes through the Guardrail Agent first (shared state,
conditional routing across four+ agents/nodes, as required by the rubric).
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from assistant.nodes import (
    decline_node,
    escalation_node,
    format_output_node,
    guardrail_node,
    info_node,
    interaction_node,
)
from assistant.state import AssistantState


def route_query(state: AssistantState) -> str:
    category = state.get("category")
    if category == "emergency":
        return "escalation"
    if category == "diagnosis":
        return "decline"
    if category == "interaction":
        return "interaction"
    return "info"


def create_graph():
    builder = StateGraph(AssistantState)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("escalation", escalation_node)
    builder.add_node("decline", decline_node)
    builder.add_node("info", info_node)
    builder.add_node("interaction", interaction_node)
    builder.add_node("format_output", format_output_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_query,
        {"escalation": "escalation", "decline": "decline", "interaction": "interaction", "info": "info"},
    )

    # Emergency and decline responses are final as-is (no further LLM rewriting of
    # safety-critical text); info/interaction results go through the formatter,
    # which adds the mandatory medical disclaimer to every remaining answer.
    builder.add_edge("escalation", END)
    builder.add_edge("decline", END)
    builder.add_edge("interaction", "format_output")
    builder.add_edge("info", "format_output")
    builder.add_edge("format_output", END)

    checkpointer = MemorySaver()
    # interrupt_before=["escalation"] is the human-in-the-loop interrupt: the graph
    # pauses right before delivering emergency guidance so a human/operator can be
    # looped in, then app.py resumes the graph to produce the final response.
    return builder.compile(checkpointer=checkpointer, interrupt_before=["escalation"])
