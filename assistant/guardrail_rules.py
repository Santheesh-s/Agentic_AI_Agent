"""Deterministic, rule-based safety net for the Guardrail Agent.

The reference architecture classifies every query with a single LLM call.
That is risky for a safety-critical category like "emergency": if the LLM
mis-reads the message, an emergency could silently fall through to a normal
info answer. This module adds a fast, offline pattern layer that runs BEFORE
the LLM and can only make the outcome MORE cautious (escalate/decline) - it
never downgrades a real emergency, and it also lets eval/run_eval.py and the
app work even with no API key configured.
"""
import re

EMERGENCY_PATTERNS = [
    r"chest (pain|pressure|tightness)",
    r"pain.{0,20}(arm|jaw|shoulder|neck|back).{0,10}(numb|heavy|weak)",
    r"(arm|face).{0,15}(numb|drooping|droops|weak(ness)?)",
    r"can'?t breathe|cannot breathe|difficulty breathing|trouble breathing|struggling to breathe",
    r"lips?.{0,10}(blue|turning blue)",
    r"slurred speech|speech.{0,10}slurred|can'?t speak clearly",
    r"sudden(ly)?.{0,20}(confusion|numbness|weakness|trouble (speaking|seeing|walking))",
    r"(severe|heavy|uncontrolled).{0,10}bleed(ing)?|bleeding.{0,10}(won'?t|will not) stop",
    r"vomit(ing)? blood|coughing up blood",
    r"unconscious|unresponsive|passed out|fainted|collapsed",
    r"seizure|convulsion",
    r"(throat|tongue|face).{0,15}(swelling|swollen)",
    r"anaphyla",
    r"took (way )?too many .{0,20}(pills|tablets|medicine|medication)|overdose|swallowed a whole (bottle|strip|packet)",
    r"suicid|kill myself|end my life|hurt(ing)? myself|self[- ]?harm",
    r"choking|can'?t breathe.{0,20}food",
    r"having a heart attack|think.{0,15}heart attack|having a stroke|think.{0,15}stroke",
]
_EMERGENCY_RE = [re.compile(p, re.I) for p in EMERGENCY_PATTERNS]

DIAGNOSIS_PATTERNS = [
    r"do i have\b",
    r"what('?s| is) wrong with me",
    r"diagnos(e|is)",
    r"am i (sick|ill|having)",
    r"could (it|this) be (cancer|diabetes|covid|an? \w+)",
    r"what disease do i have",
]
_DIAGNOSIS_RE = [re.compile(p, re.I) for p in DIAGNOSIS_PATTERNS]

PRESCRIPTION_PATTERNS = [
    r"what (dose|dosage) .{0,30}(should|do) i take",
    r"how (much|many) (should i|do i) take",
    r"prescribe",
    r"which medicine should i take",
    r"should i (stop|start) taking my",
    r"what should i take\b",
    r"what should (i|we) (give|do) for",
]
_PRESCRIPTION_RE = [re.compile(p, re.I) for p in PRESCRIPTION_PATTERNS]


def check_emergency(text: str) -> str | None:
    for pat in _EMERGENCY_RE:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def check_diagnosis_or_prescription(text: str) -> tuple[str | None, str | None]:
    """Returns (category, matched_text): category is 'diagnosis', 'prescription' or None."""
    for pat in _PRESCRIPTION_RE:
        m = pat.search(text)
        if m:
            return "diagnosis", m.group(0)   # graph treats both as the same decline route
    for pat in _DIAGNOSIS_RE:
        m = pat.search(text)
        if m:
            return "diagnosis", m.group(0)
    return None, None


INTERACTION_CUES = re.compile(
    r"\binteract|together|combin|\bwith\b.{0,20}(my|other|another).{0,15}(medic|drug|pill|tablet)", re.I)


def looks_like_interaction_query(text: str) -> bool:
    return bool(INTERACTION_CUES.search(text))
