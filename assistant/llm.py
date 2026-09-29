"""LLM provider selection: Groq if GROQ_API_KEY is set, else Gemini, else None.

Returning None (rather than raising) lets the guardrail's rule-based safety
net and the offline parts of the app keep working with no API key configured,
which is important for grading/evaluation without secrets.
"""
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_llm():
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
    return None


def extract_text_content(content) -> str:
    """Safely convert LLM message content (string or list of content blocks/dicts) to string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
            elif hasattr(part, "text"):
                parts.append(str(part.text))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content) if content is not None else ""


