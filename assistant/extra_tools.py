"""Two additional LangChain tools, beyond the three MCP-derived tools in
mcp_server/tools.py, as required by the course rubric.

- web_search_tool     : Tavily search restricted to trusted health domains,
                        for current advisories, outbreak news and recalls.
- otc_dosage_limits   : looks up the FACTUAL over-the-counter label maximum
                        dose for a medicine (not a personalised recommendation
                        - it never tells the user what THEY should take).
"""
import json
import os

from langchain_core.tools import tool

TRUSTED_DOMAINS = ["who.int", "fda.gov", "cdc.gov", "nhs.uk", "medlineplus.gov", "ema.europa.eu"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DRUGS_FILE = os.path.join(DATA_DIR, "mock_drugs.json")


def _load_drugs() -> list[dict]:
    if os.path.exists(DRUGS_FILE):
        with open(DRUGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@tool
def web_search_tool(query: str) -> str:
    """Searches the live web (Tavily, restricted to WHO/FDA/CDC/NHS/MedlinePlus/EMA) for
    current health advisories, disease outbreak news or drug recall notices. Only useful
    for questions about CURRENT events; use drug_info for a medicine's general facts."""
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is unavailable: TAVILY_API_KEY is not configured."
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=3, include_domains=TRUSTED_DOMAINS)
        results = search.invoke({"query": query})
        if not results:
            return "No current web results found from trusted health sources for this query."
        return "\n".join(f"- {r.get('url', '')}: {r.get('content', '')[:300]}" for r in results)
    except Exception as exc:  # network issues must never crash the assistant
        return f"Web search failed ({exc.__class__.__name__}); continuing without it."


@tool
def otc_dosage_limits(medicine_name: str) -> str:
    """Looks up the FACTUAL over-the-counter (OTC) label maximum dose for a medicine, e.g.
    'the label says up to 4000 mg per 24 hours'. This states the printed label limit only;
    it is never a personal recommendation of how much the user should take."""
    drugs = _load_drugs()
    name = medicine_name.strip().lower()
    for d in drugs:
        if d["branded_name"].lower() == name or d["generic_name"].lower() == name:
            if d.get("otc"):
                return (f"{d['branded_name']} ({d['generic_name']}) OTC label maximum: "
                        f"{d['max_daily_dose']}. This is the printed label limit, not personal advice; "
                        f"a doctor or pharmacist can advise what is right for you.")
            return f"{d['branded_name']} ({d['generic_name']}) is prescription-only: {d['max_daily_dose']}"
    return f"No label dosage information found for: {medicine_name}"
