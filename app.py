import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from assistant.graph import create_graph

load_dotenv()

st.set_page_config(page_title="Medicine & Symptom Assistant", page_icon="\U0001F48A")

st.title("\U0001F48A Medicine & Symptom Assistant")
st.markdown(
    "Ask about medicines, generic alternatives, and interactions.\n\n"
    "*Note: This assistant provides informational content only and cannot diagnose or prescribe.*"
)

if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "No GROQ_API_KEY or GOOGLE_API_KEY found in your .env file. The assistant will still "
        "run using its rule-based safety net and direct dataset/knowledge-base lookups, but "
        "answers will not be rewritten in plain language by an LLM. Add a key for the full "
        "experience."
    )

if "graph" not in st.session_state:
    st.session_state.graph = create_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g., Can I take ibuprofen with my blood pressure medication?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            initial_state = {"query": prompt, "gathered_info": []}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            for _ in st.session_state.graph.stream(initial_state, config=config):
                pass  # drive the graph forward; the final state is read below

            final_state = st.session_state.graph.get_state(config)

            if "escalation" in final_state.next:
                # The graph paused before the Escalation node (human-in-the-loop interrupt).
                # In a full deployment a human operator could review the category/trigger
                # here before resuming; for this app we resume immediately.
                st.session_state.graph.invoke(None, config=config)
                final_state = st.session_state.graph.get_state(config)

            response = final_state.values.get("response", "I'm sorry, I couldn't generate a response.")
            category = final_state.values.get("category", "unknown")
            st.markdown(response)
            if category == "emergency":
                st.error("Guardrail Agent classification: EMERGENCY - normal flow was stopped.")

    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "- **Guardrail Agent** screens every message for emergencies and out-of-scope requests.\n"
        "- **Info Agent** is a ReAct agent (tool-calling loop, capped iterations) using MCP-derived "
        "tools, two extra tools, and a cited RAG knowledge base.\n"
        "- **Interaction Agent** checks known interactions, including drug-class phrases like "
        "'blood pressure medication'.\n"
        "- **Escalation Node** is a LangGraph human-in-the-loop interrupt."
    )
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(int(st.session_state.thread_id) + 1)
        st.rerun()
