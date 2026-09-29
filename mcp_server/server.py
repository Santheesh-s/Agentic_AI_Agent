"""Custom MCP server for the Medicine & Symptom Information Assistant.

Exposes three tools over the Model Context Protocol (stdio transport):
  - drug_info          : uses, side effects and warnings for a medicine
  - check_interaction  : known interactions between two or more medicines
                         (also understands drug-class phrases such as
                         "blood pressure medication")
  - find_generic       : the generic (active-ingredient) equivalent of a brand

Run standalone for testing:  python -m mcp_server.server
The LangGraph app does not talk to this process directly; it uses the
identical logic exposed as LangChain tools in mcp_server/tools.py (see the
note there on why), which is the pattern recommended for LangGraph runtimes.
"""
import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Medicine Assistant Tools")

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
    """Resolve a user term to either a single drug or a drug class (a group of drugs)."""
    term_norm = term.strip().lower()
    drug = _find_drug(term_norm, drugs)
    if drug:
        return {"kind": "drug", "drug": drug}
    if term_norm in classes:
        class_ids = classes[term_norm]
        members = [d for d in drugs if d["drug_class"] in class_ids]
        return {"kind": "class", "name": term_norm, "members": members}
    return {"kind": "unknown", "name": term}


@mcp.tool()
def drug_info(medicine_name: str) -> str:
    """Returns uses, common side effects and warnings for a medicine (by brand or generic name)."""
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


@mcp.tool()
def check_interaction(medications: list[str]) -> str:
    """Checks known interactions between two or more medicines. Also accepts a drug-class
    phrase such as 'blood pressure medication' in place of a specific medicine name."""
    drugs, classes = load_drugs(), load_classes()
    resolved = [_resolve_term(m, drugs, classes) for m in medications]
    unknown = [r["name"] for r in resolved if r["kind"] == "unknown"]

    def names(r: dict) -> list[str]:
        if r["kind"] == "drug":
            return [r["drug"]["generic_name"], r["drug"]["branded_name"].lower()]
        return [d["generic_name"] for d in r["members"]]

    def drugs_of(r: dict) -> list[dict]:
        return [r["drug"]] if r["kind"] == "drug" else r["members"]

    found = []
    known = [r for r in resolved if r["kind"] != "unknown"]
    for i in range(len(known)):
        for j in range(i + 1, len(known)):
            a, b = known[i], known[j]
            b_names = {n.lower() for n in names(b)}
            for da in drugs_of(a):
                if any(inter.lower() in b_names for inter in da.get("interactions", [])):
                    label_a = a["drug"]["branded_name"] if a["kind"] == "drug" else a["name"]
                    label_b = b["drug"]["branded_name"] if b["kind"] == "drug" else b["name"]
                    found.append(f"Potential interaction: {da['branded_name']} ({label_a}) + {label_b} - "
                                 f"check with a doctor or pharmacist before combining.")
    result = list(dict.fromkeys(found))
    out = []
    if result:
        out.append("\n".join(result))
    else:
        out.append("No known interaction found in the database between: " + ", ".join(medications) + ".")
    if unknown:
        out.append("Not found in the database: " + ", ".join(unknown) + ".")
    return "\n".join(out)


@mcp.tool()
def find_generic(branded_name: str) -> str:
    """Returns the generic (active-ingredient) equivalent for a branded medicine."""
    drugs = load_drugs()
    drug = _find_drug(branded_name, drugs)
    if drug and drug["branded_name"].lower() == branded_name.strip().lower():
        others = [d["branded_name"] for d in drugs
                  if d["generic_name"] == drug["generic_name"] and d["branded_name"] != drug["branded_name"]]
        extra = f" Other brands with the same ingredient: {', '.join(others)}." if others else ""
        return f"The generic equivalent for {branded_name} is {drug['generic_name']}.{extra}"
    if drug:  # a generic name was given
        return f"'{branded_name}' is already a generic (active ingredient) name."
    return f"Generic equivalent not found for: {branded_name}"


if __name__ == "__main__":
    mcp.run()
