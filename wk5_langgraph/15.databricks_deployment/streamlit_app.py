"""
Streamlit Chat — Deployed LangGraph Agent

A minimal chat UI that talks to the CS4603 LangGraph agent running as a
Databricks Model Serving endpoint. The endpoint is OpenAI chat-completions
compatible, so we call it with the standard `openai.OpenAI` client pointed at
`{host}/serving-endpoints` — exactly like the rest of the course.

Prerequisites:
  - The agent is deployed (see deploy_setup.py / deploy_setup.sh) and the
    endpoint is in the READY state.
  - A .env file (repo root) or the sidebar provides the host, token, and
    endpoint name of the workspace where the endpoint lives.

Run (from repo root, with the venv active):
    streamlit run wk5_langgraph/11.databricks_deployment/streamlit_app.py
"""

import os

import openai
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ENDPOINT = "cs4603-langgraph-agent"


def _default(*names: str) -> str:
    """Return the first non-empty environment variable among names."""
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def get_client(host: str, token: str) -> openai.OpenAI:
    """Build an OpenAI client pointed at the Databricks serving endpoints."""
    return openai.OpenAI(
        api_key=token,
        base_url=f"{host.rstrip('/')}/serving-endpoints",
        timeout=120,  # scale-to-zero endpoints can be slow on first (cold) call
    )


def get_endpoint_status(host: str, token: str, endpoint: str) -> tuple[str, str]:
    """Query the serving endpoint state via the Databricks REST API.

    Returns (ready, detail) where ready is READY / NOT_READY / ERROR.
    """
    url = f"{host.rstrip('/')}/api/2.0/serving-endpoints/{endpoint}"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {token}"}, timeout=15
        )
    except requests.RequestException as exc:
        return "ERROR", f"Could not reach the workspace: {exc}"

    if resp.status_code == 404:
        return "ERROR", f"Endpoint '{endpoint}' not found in this workspace."
    if resp.status_code == 401 or resp.status_code == 403:
        return "ERROR", "Unauthorized — check the host/token."
    if not resp.ok:
        return "ERROR", f"HTTP {resp.status_code}: {resp.text[:200]}"

    state = resp.json().get("state", {})
    ready = state.get("ready", "UNKNOWN")
    update = state.get("config_update", "")
    detail = f"config_update: {update}" if update else ""
    return ready, detail


def render_sidebar() -> tuple[str, str, str]:
    """Render the connection settings sidebar and return (host, token, endpoint)."""
    st.sidebar.header("Connection")
    st.sidebar.caption(
        "Point these at the workspace where the agent endpoint is deployed. "
        "Defaults are read from your .env file."
    )

    host = st.sidebar.text_input(
        "Databricks host",
        value=_default("DATABRICKS_HOST_FREE", "DATABRICKS_HOST"),
        placeholder="https://<workspace>.databricks.com",
    )
    token = st.sidebar.text_input(
        "Databricks token (PAT)",
        value=_default("DATABRICKS_TOKEN_FREE", "DATABRICKS_TOKEN"),
        type="password",
        placeholder="dapi...",
    )
    endpoint = st.sidebar.text_input("Endpoint name", value=DEFAULT_ENDPOINT)

    if st.sidebar.button("Check endpoint status", use_container_width=True):
        if not host or not token:
            st.sidebar.error("Set the host and token first.")
        else:
            with st.sidebar:
                with st.spinner("Querying endpoint…"):
                    ready, detail = get_endpoint_status(host, token, endpoint)
            if ready == "READY":
                st.sidebar.success(f"READY — {endpoint}")
            elif ready == "ERROR":
                st.sidebar.error(detail)
            else:
                st.sidebar.warning(f"{ready} — not ready yet. {detail}")

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return host, token, endpoint


def ask_agent(client: openai.OpenAI, endpoint: str, history: list[dict]) -> str:
    """Send the conversation to the serving endpoint and return the reply text."""
    response = client.chat.completions.create(
        model=endpoint,
        messages=history,
    )
    return response.choices[0].message.content


def main():
    st.set_page_config(page_title="Deployed Agent Chat", page_icon="🤖", layout="centered")
    st.title("🤖 CS4603 Deployed Agent")
    st.caption("Chat with the LangGraph agent served on Databricks Model Serving.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    host, token, endpoint = render_sidebar()

    # Replay the conversation so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask the agent…")
    if not prompt:
        return

    if not host or not token:
        st.error("Set the Databricks host and token in the sidebar first.")
        return

    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the endpoint and show the reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply = ask_agent(get_client(host, token), endpoint, st.session_state.messages)
            except openai.APIConnectionError as exc:
                st.error(
                    "Connection error — the endpoint may be cold/starting or the "
                    "host is wrong. Use **Check endpoint status** in the sidebar to "
                    f"confirm it is READY, then retry. Details: {exc}"
                )
                st.session_state.messages.pop()  # drop the unanswered user turn
                return
            except Exception as exc:  # surface auth / endpoint errors to the user
                st.error(f"Request failed: {exc}")
                st.session_state.messages.pop()  # drop the unanswered user turn
                return
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
