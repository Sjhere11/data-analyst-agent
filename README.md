# AI Data Analyst Agent

Upload a CSV, ask questions in plain English, get SQL, Python analysis and charts back. An LLM agent plans the analysis, calls tools (SQL and Python execution), checks its own numbers and answers with a visualization.

## How it works

```
User question
   -> Agent orchestrator (Claude, tool-calling loop, may re-analyze)
        -> run_sql     DuckDB query on the uploaded data
        -> run_python  pandas / plotly on the DataFrame
   -> Interpret results
   -> Answer + chart
```

- **Agent loop:** the model calls tools, reads the results and repeats (max 8 steps) before answering.
- **SQL:** the CSV is registered as a DuckDB table named `data`.
- **Python:** code runs with `df`, `pd` and `px` available; assigning a Plotly figure to `fig` renders it in the chat.
- **UI:** Streamlit chat with a data preview in the sidebar.

## Stack

Python, Streamlit, Anthropic API (tool use), DuckDB, pandas, Plotly

## Run locally

```bash
git clone https://github.com/Sjhere11/data-analyst-agent.git
cd data-analyst-agent
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key   # Windows: set ANTHROPIC_API_KEY=your_key
streamlit run app.py
```

Upload any CSV in the sidebar and start asking, for example:

- "Which category has the highest average revenue?"
- "Plot monthly sales and tell me the trend."
- "Find outliers in the price column."

## Security note

`run_python` executes model-written code with `exec` on your machine. Use it for local, trusted data only. A sandboxed executor is on the roadmap.

## Roadmap

- [ ] Sandboxed code execution
- [ ] Postgres / SQL database connector
- [ ] Multi-file / joined tables
- [ ] Export analysis as a report

## Author

Siddhant Jawade - [GitHub](https://github.com/Sjhere11)
