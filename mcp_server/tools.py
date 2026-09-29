"""LangChain tool wrappers around the MCP server's logic.

The three tools below implement the exact same logic that mcp_server/server.py
exposes over the MCP stdio protocol (drug_info, check_interaction, find_generic).
They are re-declared here as @tool-decorated LangChain callables so the
LangGraph nodes and the ReAct agent in assistant/react_agent.py can call them
directly in-process. Running a second stdio subprocess for every graph
invocation adds latency and complexity that a course project does not need;
keeping one source of truth (mcp_server/server.py) and re-using it here is the
standard way MCP tools are wired into a LangGraph app. To confirm the MCP
server itself works over stdio, run:  python -m mcp_server.server
"""
import json
import os

from langchain_core.tools import tool

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DRUGS_FILE = os.path.join(DATA_DIR, "mock_drugs.json")
CLASSES_FILE = os.path.join(DATA_DIR, "drug_classes.json")


def load_drugs() -> list[dict]:
    if os.path.exists(DRUGS_FILE):
        with open(DRUGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_classes() -> dict:
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _find_drug(name: str, drugs: list[dict]) -> dict | None:
    name = name.strip().lower()
    for d in drugs:
        if d["branded_name"].lower() == name or d["generic_name"].lower() == name:
            return d
    return None


def _resolve_term(term: str, drugs: list[dict], classes: dict) -> dict:
    term_norm = term.strip().lower()
    drug = _find_drug(term_norm, drugs)
    if drug:
        return {"kind": "drug", "drug": drug}
    if term_norm in classes:
        members = [d for d in drugs if d["drug_class"] in classes[term_norm]]
        return {"kind": "class", "name": term_norm, "members": members}
    return {"kind": "unknown", "name": term}


@tool
def drug_info(medicine_name: str) -> str:
    """Returns uses, common side effects and warnings for a medicine, by brand or generic name."""
    drugs = load_drugs()
    drug = _find_drug(medicine_name, drugs)
    if not drug:
        return f"Information not found for medicine: {medicine_name}"
    lines = [
        f"Name: {drug['branded_name']} (Generic: {drug['generic_name']}, Class: {drug['drug_class']})",
        f"Uses: {drug['uses']}",
        f"Side Effects: {drug['side_effects']}",
        f"Warnings: {drug['warnings']}",
    ]
    if drug.get("otc"):
        lines.append(f"OTC label maximum dose: {drug['max_daily_dose']}")
    return "\n".join(lines)


@tool
def check_interaction(medications_csv: str) -> str:
    """Checks known interactions between two or more medicines. Input is a comma-separated
    list, e.g. 'ibuprofen, warfarin' or 'ibuprofen, blood pressure medication'. Drug-class
    phrases such as 'blood pressure medication' or 'blood thinner' are also understood."""
    medications = [m.strip() for m in medications_csv.split(",") if m.strip()]
    drugs, classes = load_drugs(), load_classes()
    resolved = [_resolve_term(m, drugs, classes) for m in medications]
    unknown = [r["name"] for r in resolved if r["kind"] == "unknown"]
    known = [r for r in resolved if r["kind"] != "unknown"]

    def names(r: dict) -> set[str]:
        if r["kind"] == "drug":
            return {r["drug"]["generic_name"].lower(), r["drug"]["branded_name"].lower()}
        return {d["generic_name"].lower() for d in r["members"]}

    def drugs_of(r: dict) -> list[dict]:
        return [r["drug"]] if r["kind"] == "drug" else r["members"]

    found = []
    for i in range(len(known)):
        for j in range(i + 1, len(known)):
            a, b = known[i], known[j]
            b_names = names(b)
            for da in drugs_of(a):
                if any(inter.lower() in b_names for inter in da.get("interactions", [])):
                    label_a = a["drug"]["branded_name"] if a["kind"] == "drug" else a["name"]
                    label_b = b["drug"]["branded_name"] if b["kind"] == "drug" else b["name"]
                    found.append(f"Potential interaction: {da['branded_name']} ({label_a}) + {label_b} - "
                                 f"check with a doctor or pharmacist before combining.")
    result = list(dict.fromkeys(found))
    out = ["\n".join(result)] if result else \
        [f"No known interaction found in the database between: {', '.join(medications)}."]
    if unknown:
        out.append("Not found in the database: " + ", ".join(unknown) + ".")
    return "\n".join(out)


@tool
def find_generic(branded_name: str) -> str:
    """Returns the generic (active-ingredient) equivalent for a branded medicine."""
    drugs = load_drugs()
    drug = _find_drug(branded_name, drugs)
    if drug and drug["branded_name"].lower() == branded_name.strip().lower():
        others = [d["branded_name"] for d in drugs
                  if d["generic_name"] == drug["generic_name"] and d["branded_name"] != drug["branded_name"]]
        extra = f" Other brands with the same ingredient: {', '.join(others)}." if others else ""
        return f"The generic equivalent for {branded_name} is {drug['generic_name']}.{extra}"
    if drug:
        return f"'{branded_name}' is already a generic (active ingredient) name."
    return f"Generic equivalent not found for: {branded_name}"
