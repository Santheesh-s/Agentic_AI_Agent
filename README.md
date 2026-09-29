# Medicine & Symptom Information Assistant

An informational, non-diagnostic AI assistant that explains medicines, checks drug interactions,
suggests generic equivalents, and shares verified health information, while actively recognizing
emergencies and directing users to healthcare professionals.

This is Task 5 of the Agentic Systems Course capstone. It satisfies every mandatory requirement in
the Capstone Tasks and Evaluation Rubric:

| Rubric requirement | Where it lives |
|---|---|
| Custom MCP server with at least 3 tools | `mcp_server/server.py`: `drug_info`, `check_interaction`, `find_generic` (stdio, `python -m mcp_server.server`) |
| At least 2 additional LangChain tools | `assistant/extra_tools.py`: `web_search_tool` (Tavily), `otc_dosage_limits` |
| RAG pipeline with citations | `rag/retriever.py` (FAISS + HuggingFace embeddings) tags every chunk with its source file; `format_with_citations` renders `[Source: filename]` on every fact used |
| Tavily web search | `web_search_tool`, restricted to WHO/FDA/CDC/NHS/MedlinePlus/EMA |
| At least 1 ReAct agent with an iteration limit | `assistant/react_agent.py`: the Info Agent is `langgraph.prebuilt.create_react_agent`, capped via `recursion_limit` |
| LangGraph orchestration, 3+ agents, shared state, conditional routing | `assistant/graph.py`: Guardrail -> {Escalation \| Decline \| Interaction \| Info} -> Format Output, all sharing `AssistantState` |

---

## 📁 Project Architecture & Folder Structure

```
Medicine & Symptom Information Assistant/
├── app.py                     # Streamlit web UI with LangGraph interrupt escalation handling
├── requirements.txt           # Project dependencies
├── .env.example                # Copy to .env and fill in your API keys
├── README.md                  # Project documentation
├── assistant/
│   ├── state.py               # AssistantState TypedDict schema (shared graph state)
│   ├── guardrail_rules.py     # Deterministic red-flag / diagnosis / prescription pattern layer
│   ├── llm.py                 # LLM provider selection (Groq, then Gemini, else None)
│   ├── extra_tools.py         # 2 additional LangChain tools: web_search_tool, otc_dosage_limits
│   ├── react_agent.py         # ReAct Info Agent (langgraph.prebuilt) with an iteration limit
│   ├── nodes.py                # Guardrail, Escalation, Decline, Info, Interaction, Format Output nodes
│   └── graph.py                # LangGraph StateGraph: conditional routing & escalation interrupt
├── rag/
│   ├── __init__.py
│   ├── retriever.py           # FAISS vector store + HuggingFace embeddings, with source citations
│   └── knowledge_base/
│       ├── who_factsheets.txt # WHO-style fact sheets (pain relief, hypertension, AMR, diabetes, generics)
│       ├── drug_leaflets.txt  # Drug leaflets: uses, warnings, side effects, interactions
│       └── first_aid_guide.txt# First-aid emergency symptom recognition guide
├── mcp_server/
│   ├── server.py               # FastMCP server: drug_info, check_interaction, find_generic
│   └── tools.py                # Same logic as LangChain @tool functions (used by the graph/agent)
├── data/
│   ├── mock_drugs.json        # Curated dataset: 16 medicines, uses, warnings, interactions, OTC limits
│   ├── drug_classes.json       # Maps phrases like "blood pressure medication" to drug classes
│   └── who_factsheets.txt     # Baseline fact sheet used only if rag/knowledge_base is empty
└── eval/
    └── run_eval.py             # Guardrail evaluation suite for red-flag triage detection
```

---

## 🚀 Setup & Execution

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   ```bash
   cp .env.example .env
   # then edit .env and add GROQ_API_KEY or GOOGLE_API_KEY, and optionally TAVILY_API_KEY
   ```
   The assistant also runs with **no keys at all** — see "Offline mode" below.

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

5. **Run the guardrail evaluation suite:**
   ```bash
   python eval/run_eval.py
   ```

6. **(Optional) Run the MCP server standalone**, to confirm the tools work over the actual
   Model Context Protocol (stdio), separately from the in-process LangChain wrappers the graph uses:
   ```bash
   python -m mcp_server.server
   ```

---

## 🧠 How it works

Every query passes through the **Guardrail Agent** first (`assistant/nodes.py::guardrail_node`),
which classifies it into `emergency`, `diagnosis`, `interaction` or `info` and routes accordingly
(`assistant/graph.py`):

```
                     +--> Escalation  (LangGraph interrupt_before) --+
User -> Guardrail ---+--> Decline    (diagnosis / prescription) -----+--> END
                     +--> Interaction Agent ------------------------+--> Format Output --> END
                     +--> Info Agent (ReAct, iteration-limited) -----+
```

**The Guardrail Agent is two layers, not one.** A single LLM call to classify a message is what a
naive design would do — but for a safety-critical category like "emergency", a single model call
that might mis-read the phrasing is too risky. `assistant/guardrail_rules.py` runs a fast,
deterministic regex layer *before* any LLM call: it catches the clearest red flags (chest pain,
stroke signs, breathing trouble, severe bleeding, overdose, suicidal statements) and the clearest
diagnosis/prescription phrasing, with no API key required. The LLM is only consulted for the
remaining, more nuanced cases (interaction vs. info, and as a second opinion on phrasing the rules
missed) — it can make the outcome *more* cautious, never override a rule-confirmed emergency.

**The Info Agent is a real ReAct agent**, not a fixed pipeline. `assistant/react_agent.py` builds it
with `langgraph.prebuilt.create_react_agent` over five tools (the three MCP-derived tools plus the
two extra tools), and the agent decides for itself which tools to call and in what order. Its
iteration limit is enforced via `recursion_limit` on `agent.invoke(...)` (`MAX_ITERATIONS = 5`
tool-calling steps): if the agent hasn't converged by then, it must answer with whatever it has
gathered rather than loop indefinitely.

**The RAG pipeline cites its sources.** Every chunk in the FAISS index is tagged with its source
filename at load time (`rag/retriever.py`), and `format_with_citations` renders `[Source: filename]`
next to every fact pulled from the knowledge base, which the formatter prompt is instructed to keep.

**Interactions understand drug classes**, not just exact names: "Can I take ibuprofen with my
blood pressure medication?" resolves the phrase via `data/drug_classes.json` to the five blood
pressure drug classes in the dataset (ACE inhibitor, ARB, diuretic, calcium channel blocker,
beta-blocker) and checks each member for a known interaction with ibuprofen.

### Offline mode (no API keys)
With no `GROQ_API_KEY`/`GOOGLE_API_KEY` configured, `assistant/llm.py::get_llm()` returns `None` and
every node degrades gracefully instead of crashing: the guardrail's rule layer still classifies
every query (94.7% accuracy, **100% emergency recall**, on `eval/run_eval.py`'s 19-query set), the
Info Agent falls back to direct dataset/knowledge-base lookups, and the Interaction Agent extracts
medicines by matching dataset names/classes as whole words in the query. This is intentional: a
grader without your API keys should still see the safety-critical parts of the system work.

---

## 🛡️ Guardrails and Constraints
- **Emergency escalation:** Red-flag symptoms (chest pain, stroke signs, difficulty breathing,
  severe bleeding, overdose, suicidal statements, and more) trigger a LangGraph human-in-the-loop
  interrupt (`interrupt_before=["escalation"]`) and return urgent emergency-contact guidance.
- **Non-diagnostic constraint:** Requests for a diagnosis or a personal dose/prescription are
  politely declined, both by the rule layer and the LLM classifier.
- **Medical disclaimer:** Appended to every info/interaction response by `format_output_node`.
- **No invented facts:** The formatter prompt is restricted to the gathered evidence only, and the
  ReAct agent's system prompt forbids inventing facts, diagnosing, or stating a personal dose.

## 📊 Evaluation
`eval/run_eval.py` runs 19 labelled queries (8 emergencies, 4 diagnosis/prescription, 3
interactions, 4 info) through the Guardrail Agent and reports overall accuracy and emergency-only
recall. Current result with no API key (rules only): **18/19 (94.7%) overall, 8/8 (100%) emergency
detection.** The one miss is an intentionally out-of-scope case for the rule layer ("Is this rash
eczema or psoriasis?" has no red-flag phrasing) that the LLM layer resolves when a key is set —
run `python eval/run_eval.py` after adding a key to see the combined result.

## Known limitations
- The medicine dataset (`data/mock_drugs.json`) is a small curated set of 16 common medicines for
  demonstration, not a clinical database; "no known interaction found" is never a guarantee of
  safety. Swap in OpenFDA or another live source for production use.
- The regex safety net is English-only and intentionally narrow (it targets the clearest,
  highest-stakes phrasing); unusual phrasing of a genuine emergency could still need the LLM layer,
  which is why an LLM key is strongly recommended even though the app runs without one.
- `sentence-transformers`/`faiss-cpu` download a small embedding model on first run and need
  internet access; `rag/retriever.py` catches the resulting `ImportError`/runtime errors and the
  app continues without RAG citations rather than crashing if they are unavailable.
