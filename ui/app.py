"""Streamlit chat UI for the Veltrix PeopleOps Assistant."""

import uuid
import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

EMPLOYEES = {
    "E1001": "Alice Smith (India · Employee)",
    "E1002": "Bob Jones (USA · Employee)",
    "E1003": "Clara Muller (Germany · Employee)",
    "E2001": "Dan Manager (India · Manager)",
    "E2002": "Eva Manager (Germany · Manager)",
    "E3001": "HR Hannah (India · HR)",
}

st.set_page_config(page_title="Veltrix PeopleOps Assistant", page_icon="🏢")

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "employee_id" not in st.session_state:
    st.session_state.employee_id = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    options = ["— Select Employee —"] + [f"{eid} · {name}" for eid, name in EMPLOYEES.items()]
    selected = st.selectbox("Employee", options)

    if selected != "— Select Employee —":
        emp_id = selected.split(" · ")[0]
    else:
        emp_id = ""

    if emp_id != st.session_state.employee_id:
        st.session_state.employee_id = emp_id
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.success("API: Ready")
        else:
            st.warning("API: Starting…")
    except Exception:
        st.error("API: Unreachable")

# ── Title with employee name ──────────────────────────────────────────────────
if st.session_state.employee_id:
    emp_name = EMPLOYEES.get(st.session_state.employee_id, st.session_state.employee_id)
    st.title(f"🏢 Veltrix PeopleOps")
    st.subheader(f"Chatting as: {st.session_state.employee_id} — {emp_name}")
else:
    st.title("🏢 Veltrix PeopleOps Assistant")
    st.info("Select an employee from the sidebar to start chatting.")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Citations"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['doc_id']}** — {c['locator']}")
        if msg.get("tool_calls"):
            with st.expander("Tool calls"):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['name']}({tc['args']}) → {tc['status']}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask an HR question or manage your leave…"):
    if not st.session_state.employee_id:
        st.error("Please select an Employee from the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"session_id": st.session_state.session_id, "message": prompt},
                    headers={
                        "X-Employee-Id": st.session_state.employee_id,
                        "Content-Type": "application/json",
                    },
                    timeout=60,
                )

                if resp.status_code == 401:
                    st.error("Employee not found. Check your Employee ID.")
                    st.stop()
                elif resp.status_code == 403:
                    st.error("Session conflict. Starting a new session.")
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.rerun()
                elif resp.status_code != 200:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    st.stop()

                data = resp.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                tool_calls = data.get("tool_calls", [])
                route = data.get("route", "")
                pending = data.get("pending_confirmation", False)

                st.markdown(answer)
                if pending:
                    st.info("⚠️ Awaiting your confirmation. Reply **yes** to proceed.")
                if citations:
                    with st.expander("Citations"):
                        for c in citations:
                            st.markdown(f"- **{c['doc_id']}** — {c['locator']}")
                if tool_calls:
                    with st.expander("Tool calls"):
                        for tc in tool_calls:
                            st.code(f"{tc['name']}({tc['args']}) → {tc['status']}")
                st.caption(f"Route: `{route}`")

            except requests.Timeout:
                st.error("Request timed out. The LLM is taking too long — try again.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": data.get("answer", ""),
        "citations": data.get("citations", []),
        "tool_calls": data.get("tool_calls", []),
    })
