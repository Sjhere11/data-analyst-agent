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
export ANTHROPIC_API_KEY=your_key   # PowerShell: $env:ANTHROPIC_API_KEY="your_key"
streamlit run app.py
```

Upload any CSV in the sidebar and start asking, for example:

- "Which category has the highest average revenue?"
- "Plot monthly sales and tell me the trend."
- "Find outliers in the price column."

## Evaluation

`evals/eval_agent.py` tests the agent on 50 questions over a 109,941-row cafe-sales table (12 simple aggregates, 15 group-bys and rankings, 12 filters, 11 multi-step questions).

- Ground-truth answers are computed with pandas, never by the model, and every ranking question has a single unambiguous answer (ties are rejected at build time).
- Each question runs on a fresh conversation with the same tools, system prompt, model and step limit as `app.py`. The loop is a copy of the one in `app.py`, because `app.py` is a Streamlit script and cannot be imported.
- The final `ANSWER:` line is graded: numbers within rounding tolerance, names exact.
- The script prints the score as k/50 with a 95% Wilson interval, average steps, tool errors and step-limit hits, and writes per-question results to `eval_results.csv`.

```bash
git clone https://github.com/Sjhere11/menuiq ../menuiq     # source tables (simulated cafe data)
export ANTHROPIC_API_KEY=your_key                          # PowerShell: $env:ANTHROPIC_API_KEY="your_key"
python evals/eval_agent.py --raw-dir ../menuiq/data/raw
python evals/eval_agent.py --raw-dir ../menuiq/data/raw --dry-run oracle   # harness self-check, no API calls
```

## Security note

`run_python` executes model-written code with `exec` on your machine. Use it for local, trusted data only. A sandboxed executor is on the roadmap.

## Roadmap

- [x] 50-question evaluation harness (`evals/`)
- [ ] Sandboxed code execution
- [ ] Postgres / SQL database connector
- [ ] Multi-file / joined tables
- [ ] Export analysis as a report

## Author

Siddhant Jawade - [GitHub](https://github.com/Sjhere11)
