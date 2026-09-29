from typing import TypedDict, Literal


class AssistantState(TypedDict, total=False):
    """Shared state that flows through every node in the LangGraph graph."""
    query: str
    category: Literal["emergency", "diagnosis", "interaction", "info", "unknown"]
    trigger: str            # which red-flag/out-of-scope pattern fired (for transparency + eval)
    gathered_info: list[str]
    response: str
