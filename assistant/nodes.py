"""The five graph nodes: Guardrail, Escalation, Decline, Info, Interaction and
the final output formatter. See assistant/graph.py for how they are wired.
"""
import json
import os
import re

from langchain_core.prompts import ChatPromptTemplate

from assistant.guardrail_rules import (
    check_diagnosis_or_prescription,
    check_emergency,
    looks_like_interaction_query,
)
from assistant.llm import extract_text_content, get_llm
from assistant.react_agent import run_info_agent
from assistant.state import AssistantState
from mcp_server.tools import check_interaction, load_classes, load_drugs
from rag import format_with_citations, setup_rag

EMERGENCY_NUMBERS = "911 (US/Canada), 112 (EU/India), 999 (UK), 000 (Australia)"




# --------------------------------------------------------------------------- #
# 1. Guardrail Agent
# --------------------------------------------------------------------------- #
def guardrail_node(state: AssistantState) -> AssistantState:
    """Screens every query for emergencies, out-of-scope requests, interactions or info.

    Layer 1 (always runs, no API key needed): deterministic red-flag and
    diagnosis/prescription pattern matching (assistant/guardrail_rules.py).
    This is a safety net an LLM classifier alone cannot guarantee - it can
    only ever push the outcome towards "more cautious", never override an
    emergency the rules already caught.
    Layer 2 (needs an LLM): asks the model to classify the remaining cases
    into interaction vs info, and as a second opinion for emergency/diagnosis
    phrasing the rules may have missed.
    """
    query = state["query"]

    matched = check_emergency(query)
    if matched:
        return {"category": "emergency", "trigger": matched}

    category, matched = check_diagnosis_or_prescription(query)
    if category:
        return {"category": "diagnosis", "trigger": matched}

    llm = get_llm()
    if llm is None:
        # Offline fallback: no API key configured. Two or more recognised medicines/drug-class
        # phrases in the same query is a strong interaction signal even without a phrasing
        # cue like "with" or "together" (e.g. "Can I take X if I am already taking Y?").
        meds = _regex_extract_medications(query)
        category = "interaction" if len(meds) >= 2 or looks_like_interaction_query(query) else "info"
        return {"category": category, "trigger": ""}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a medical triage and guardrail assistant. Classify the user query into "
         "exactly one of: 'emergency' (life-threatening symptoms: chest pain, sudden "
         "numbness/weakness, severe bleeding, difficulty breathing, stroke signs, overdose, "
         "suicidal statements), 'diagnosis' (asks you to diagnose them, or asks what dose/"
         "medicine THEY should take or whether to start/stop a medicine), 'interaction' "
         "(asks whether two or more specific medicines/medicine classes can be taken "
         "together), or 'info' (general facts about a medicine, generic equivalents, or "
         "non-emergency health information). Respond ONLY with one exact word."),
        ("human", "Query: {query}"),
    ])
    result = (prompt | llm).invoke({"query": query})
    text = extract_text_content(result.content).strip().lower()
    if "emergency" in text:
        category = "emergency"
    elif "diagnosis" in text:
        category = "diagnosis"
    elif "interaction" in text:
        category = "interaction"
    elif "info" in text:
        category = "info"
    else:
        category = "info"
    return {"category": category, "trigger": ""}


# --------------------------------------------------------------------------- #
# 2. Escalation Node (human-in-the-loop interrupt target)
# --------------------------------------------------------------------------- #
def escalation_node(state: AssistantState) -> AssistantState:
    """Returns emergency guidance. graph.py pauses the run (interrupt_before=['escalation'])
    before this node runs, so a human/operator sees the emergency category first."""
    response = (
        "\U0001F6A8 **THIS MAY BE A MEDICAL EMERGENCY** \U0001F6A8\n\n"
        f"Please call your local emergency number right now ({EMERGENCY_NUMBERS}) or go to "
        "the nearest emergency department. If someone else is affected, stay with them and "
        "do not leave them alone.\n\n"
        "I cannot help with medical emergencies - please get professional help first. "
        "You are welcome to ask me general questions afterwards."
    )
    return {"response": response}


# --------------------------------------------------------------------------- #
# 3. Decline Node (diagnosis / prescription requests)
# --------------------------------------------------------------------------- #
def decline_node(state: AssistantState) -> AssistantState:
    response = (
        "I can't diagnose conditions or tell you what medicine or dose to take - only a "
        "doctor or pharmacist who knows your health history can safely do that.\n\n"
        "What I can help with: explaining what a medicine is generally used for and its "
        "common side effects and warnings, checking known interactions between medicines, "
        "suggesting generic equivalents, and sharing general health information."
    )
    return {"response": response}


# --------------------------------------------------------------------------- #
# 4. Info Agent (ReAct agent with an iteration limit; RAG with citations)
# --------------------------------------------------------------------------- #
def info_node(state: AssistantState) -> AssistantState:
    query = state["query"]
    gathered: list[str] = []

    retriever = setup_rag()
    if retriever:
        try:
            docs = retriever.invoke(query)
            cited = format_with_citations(docs)
            if cited:
                gathered.append("Knowledge base (WHO fact sheets / leaflets / first-aid guide):\n" + cited)
        except Exception as exc:
            gathered.append(f"(Knowledge base search unavailable: {exc.__class__.__name__})")

    llm = get_llm()
    if llm is not None:
        agent_answer = run_info_agent(llm, query)
        gathered.append("Info Agent (tools used as needed - drug_info / find_generic / "
                        "check_interaction / otc_dosage_limits / web_search_tool):\n" + agent_answer)
    else:
        # Offline fallback: try a direct dataset lookup for any drug name mentioned.
        drugs = load_drugs()
        mentioned = [d for d in drugs if d["branded_name"].lower() in query.lower()
                    or d["generic_name"].lower() in query.lower()]
        for d in mentioned[:3]:
            gathered.append(f"Database info for {d['branded_name']} (Generic: {d['generic_name']}):\n"
                            f"Uses: {d['uses']}\nSide Effects: {d['side_effects']}\nWarnings: {d['warnings']}")
        if not mentioned and not gathered:
            gathered.append("No LLM is configured and no matching medicine was found in the dataset "
                            "for this query.")

    return {"gathered_info": gathered}


# --------------------------------------------------------------------------- #
# 5. Interaction Agent
# --------------------------------------------------------------------------- #
_MED_SPLIT = re.compile(r",| and |\bwith\b", re.I)


def _word_in(term: str, text: str) -> bool:
    """Whole-word/phrase match so 'statin' doesn't falsely match inside 'atorvastatin'."""
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _regex_extract_medications(query: str) -> list[str]:
    """Offline fallback: match dataset drug names/classes that appear in the query as whole words."""
    drugs, classes = load_drugs(), load_classes()
    found = []
    low = query.lower()
    for d in drugs:
        if _word_in(d["branded_name"].lower(), low) or _word_in(d["generic_name"].lower(), low):
            found.append(d["generic_name"])
    for phrase in classes:
        if _word_in(phrase, low):
            found.append(phrase)
    return list(dict.fromkeys(found))


def interaction_node(state: AssistantState) -> AssistantState:
    query = state["query"]
    llm = get_llm()

    if llm is not None:
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract all medication names or drug-class phrases (e.g. 'blood pressure "
             "medication', 'blood thinner') from the query. Respond with a comma-separated "
             "list only, e.g. 'ibuprofen, blood pressure medication'."),
            ("human", "{query}"),
        ])
        raw_content = (prompt | llm).invoke({"query": query}).content
        meds_csv = extract_text_content(raw_content).strip()
    else:
        meds = _regex_extract_medications(query)
        meds_csv = ", ".join(meds)

    if not meds_csv or meds_csv.upper() == "NONE":
        return {"gathered_info": ["Could not identify at least two medicines in the query to check."]}

    result = check_interaction.invoke({"medications_csv": meds_csv})
    return {"gathered_info": [f"Interaction Check Results (checked: {meds_csv}):\n{result}"]}


# --------------------------------------------------------------------------- #
# 6. Output Formatter (adds the mandatory medical disclaimer to every answer)
# --------------------------------------------------------------------------- #
DISCLAIMER = (
    "\n\n---\n**Medical disclaimer:** This information is for educational purposes only and "
    "is not a substitute for professional medical advice, diagnosis or treatment. Always "
    "consult your doctor or pharmacist about your own situation."
)


def format_output_node(state: AssistantState) -> AssistantState:
    llm = get_llm()
    gathered_info = state.get("gathered_info", [])
    info_str = "\n\n".join(gathered_info)

    if llm is None:
        # Offline: present the gathered evidence directly rather than paraphrasing it.
        body = info_str if info_str else "I could not find information to answer that."
        return {"response": body + DISCLAIMER}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful, plain-language medical information assistant. Using ONLY the "
         "provided gathered information, answer the user's query clearly and simply. Keep "
         "any [Source: ...] citations from the knowledge base. Do not invent information, "
         "diagnose, prescribe or state a personal dose.\n\nGathered Info:\n{gathered_info}"),
        ("human", "{query}"),
    ])
    result = (prompt | llm).invoke({"query": state["query"], "gathered_info": info_str})
    return {"response": extract_text_content(result.content) + DISCLAIMER}
