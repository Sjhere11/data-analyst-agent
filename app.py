import io
import contextlib

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"
MAX_STEPS = 8
client = Anthropic()  # reads ANTHROPIC_API_KEY

TOOLS = [
    {
        "name": "run_sql",
        "description": "Run a DuckDB SQL query on table `data`. Returns up to 50 rows.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run Python. Available: df (pandas), pd, px (plotly.express). "
            "print() to return output. Assign a plotly figure to `fig` to show a chart."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]


def run_tool(name, args, df, figs):
    try:
        if name == "run_sql":
            con = duckdb.connect()
            con.register("data", df)
            return con.execute(args["query"]).df().head(50).to_string()
        if name == "run_python":
            ns = {"df": df.copy(), "pd": pd, "px": px}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(args["code"], ns)
            if "fig" in ns:
                figs.append(ns["fig"])
            return (buf.getvalue() or "ok")[:4000]
    except Exception as e:
        return f"ERROR: {e}"


def ask(question, df):
    system = (
        "You are a data analyst agent. Table `data` / DataFrame `df` columns:\n"
        f"{df.dtypes.to_string()}\n\nSample:\n{df.head(5).to_string()}\n\n"
        "Use tools to answer. Verify numbers before stating them. Be concise. "
        "Make a chart when it helps."
    )
    msgs = st.session_state.api_msgs
    msgs.append({"role": "user", "content": question})
    figs = []
    for _ in range(MAX_STEPS):
        r = client.messages.create(
            model=MODEL, max_tokens=2000, system=system, tools=TOOLS, messages=msgs
        )
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), figs
        results = [
            {
                "type": "tool_result",
                "tool_use_id": b.id,
                "content": run_tool(b.name, b.input, df, figs),
            }
            for b in r.content
            if b.type == "tool_use"
        ]
        msgs.append({"role": "user", "content": results})
    return "Hit step limit.", figs


st.title("AI Data Analyst Agent")
st.session_state.setdefault("api_msgs", [])
st.session_state.setdefault("shown", [])

file = st.sidebar.file_uploader("Upload CSV", type="csv")
if not file:
    st.info("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(file)
st.sidebar.write(df.shape)
st.sidebar.dataframe(df.head())

for role, text, figs in st.session_state.shown:
    with st.chat_message(role):
        st.write(text)
        for f in figs:
            st.plotly_chart(f, use_container_width=True)

if q := st.chat_input("Ask about your data"):
    st.session_state.shown.append(("user", q, []))
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            text, figs = ask(q, df)
        st.write(text)
        for f in figs:
            st.plotly_chart(f, use_container_width=True)
    st.session_state.shown.append(("assistant", text, figs))
