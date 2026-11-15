#!/usr/bin/env python3
"""50-question eval for the AI Data Analyst Agent (github.com/Sjhere11/data-analyst-agent).

How it works
  * Builds one flat 109,941-row table from the MenuIQ raw CSVs (order lines + item + order info).
  * 50 questions, each with a ground-truth answer computed by pandas (never by the model).
  * Runs the SAME agent loop as app.py (same tools, system prompt, model, MAX_STEPS) once per question,
    each on a fresh conversation, and grades the final `ANSWER:` line against the ground truth.
  * Prints the score as k/50 with a 95% Wilson interval, plus step and tool-error stats.

Setup
  git clone https://github.com/Sjhere11/menuiq ../menuiq   # provides the CSV tables
  pip install anthropic duckdb pandas plotly
  export ANTHROPIC_API_KEY=...                          # PowerShell: $env:ANTHROPIC_API_KEY="..."

Run
  python evals/eval_agent.py --raw-dir ../menuiq/data/raw             # real run
  python evals/eval_agent.py --raw-dir ../menuiq/data/raw --dry-run oracle   # harness check, no API calls

Note: run_python executes model-written code with exec (same as the app). Local, trusted data only.
"""
import argparse
import contextlib
import csv
import io
import math
import re
import sys

import duckdb
import pandas as pd
import plotly.express as px

MODEL = "claude-sonnet-5-5"  # same as app.py
MAX_STEPS = 8                # same as app.py
SUFFIX = ("\n\nFinish with one last line formatted exactly as `ANSWER: <value>` "
          "(a plain number with no commas, currency or % sign, or a plain name).")

TOOLS = [
    {"name": "run_sql", "description": "Run a DuckDB SQL query on table `data`. Returns up to 50 rows.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "run_python",
     "description": ("Run Python. Available: df (pandas), pd, px (plotly.express). "
                     "print() to return output. Assign a plotly figure to `fig` to show a chart."),
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
]


# ---------------------------------------------------------------- data + ground truth
def build_table(raw_dir):
    o = pd.read_csv(f"{raw_dir}/fact_orders.csv")
    i = pd.read_csv(f"{raw_dir}/fact_order_items.csv")
    d = pd.read_csv(f"{raw_dir}/dim_item.csv")[["item_id", "item_name", "category"]]
    df = i.merge(d, on="item_id").merge(
        o[["order_id", "order_date", "order_hour", "channel", "payment"]], on="order_id")
    df["revenue"] = df.qty * df.unit_price
    df["profit"] = df.qty * (df.unit_price - df.unit_cost)
    dt = pd.to_datetime(df.order_date)
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.day_name()
    cols = ["order_id", "order_date", "month", "weekday", "order_hour", "channel", "payment",
            "item_name", "category", "qty", "unit_price", "unit_cost", "revenue", "profit"]
    return df[cols]


def rank(s, n=1, largest=True):
    """Label at rank n; asserts the rank is not tied so every question has one right answer."""
    s = s.sort_values(ascending=not largest)
    for j in (n - 2, n):
        if 0 <= j < len(s):
            assert s.iloc[j] != s.iloc[n - 1], f"tie at rank {n}"
    return s.index[n - 1]


def build_questions(df):
    Q = []

    def add(text, truth, kind="num", d=0):
        Q.append(dict(id=len(Q) + 1, q=text, truth=truth, kind=kind, d=d))

    R, P = df.revenue, df.profit
    orders = lambda x: x.order_id.nunique()
    n_orders = orders(df)
    o_by = lambda col: df.groupby(col).order_id.nunique()
    rev_by = lambda col: df.groupby(col).revenue.sum()
    per_order = df.groupby("order_id").revenue.sum()

    # --- 1-12 simple aggregates
    add("How many rows does the table have?", len(df))
    add("How many distinct orders (order_id) are there?", n_orders)
    add("What is the total revenue (sum of the revenue column), rounded to 0 decimals?", R.sum())
    add("What is the total profit (sum of the profit column), rounded to 0 decimals?", P.sum())
    add("What is the total number of units sold (sum of qty)?", df.qty.sum())
    add("What is the average unit_price across all rows, rounded to 2 decimals?", df.unit_price.mean(), d=2)
    add("What is the maximum unit_price in the table?", df.unit_price.max(), d=0)
    add("How many distinct item_name values are there?", df.item_name.nunique())
    add("How many distinct categories are there?", df.category.nunique())
    add("What is the overall gross margin percentage (total profit / total revenue * 100), rounded to 1 decimal?",
        P.sum() / R.sum() * 100, d=1)
    add("What is the average qty per row, rounded to 2 decimals?", df.qty.mean(), d=2)
    add("What is the average order value (total revenue / distinct orders), rounded to 2 decimals?",
        R.sum() / n_orders, d=2)

    # --- 13-27 group-bys and rankings
    add("Which channel has the most distinct orders?", rank(o_by("channel")), "str")
    add("Which payment method is used on the most distinct orders?", rank(o_by("payment")), "str")
    add("Which category has the highest total revenue?", rank(rev_by("category")), "str")
    add("Which category has the lowest total revenue?", rank(rev_by("category"), largest=False), "str")
    add("Which item_name sold the most units (sum of qty)?", rank(df.groupby("item_name").qty.sum()), "str")
    add("Which item_name has the highest total profit?", rank(df.groupby("item_name").profit.sum()), "str")
    add("Which month number (1-12) has the highest total revenue?", rank(rev_by("month")))
    add("Which month number (1-12) has the fewest distinct orders?", rank(o_by("month"), largest=False))
    add("Which weekday (full English name) has the most distinct orders?", rank(o_by("weekday")), "str")
    add("Which order_hour (0-23) has the most distinct orders?", rank(o_by("order_hour")))
    add("How many distinct item_name values are in the Coffee category?",
        df[df.category == "Coffee"].item_name.nunique())
    add("What is the total revenue of the Dine-in channel, rounded to 0 decimals?",
        df[df.channel == "Dine-in"].revenue.sum())
    add("How many distinct orders were paid with UPI?", orders(df[df.payment == "UPI"]))
    de = df[df.channel == "Delivery"]
    add("What is the average order value (revenue / distinct orders) for the Delivery channel, rounded to 2 decimals?",
        de.revenue.sum() / orders(de), d=2)
    add("Which item_name has the highest average unit_price?", rank(df.groupby("item_name").unit_price.mean()), "str")

    # --- 28-39 filters and conditions
    add("What is the total revenue of the Coffee category in month 3, rounded to 0 decimals?",
        df[(df.category == "Coffee") & (df.month == 3)].revenue.sum())
    add("How many units of 'Masala Chai' were sold in total?", df[df.item_name == "Masala Chai"].qty.sum())
    add("How many distinct orders were placed on a Saturday or Sunday?",
        orders(df[df.weekday.isin(["Saturday", "Sunday"])]))
    add("What is the total revenue from rows with order_hour below 12, rounded to 0 decimals?",
        df[df.order_hour < 12].revenue.sum())
    add("What is the total profit of the Takeaway channel, rounded to 0 decimals?",
        df[df.channel == "Takeaway"].profit.sum())
    add("How many rows have qty of 2 or more?", (df.qty >= 2).sum())
    add("What is the total revenue in months 7 to 12 inclusive, rounded to 0 decimals?",
        df[df.month >= 7].revenue.sum())
    add("How many distinct orders were placed in month 12?", orders(df[df.month == 12]))
    add("What is the average unit_price of 'Cold Coffee' rows in months 1 to 6, rounded to 2 decimals?",
        df[(df.item_name == "Cold Coffee") & (df.month <= 6)].unit_price.mean(), d=2)
    add("What percentage of total revenue comes from the Delivery channel, rounded to 1 decimal?",
        de.revenue.sum() / R.sum() * 100, d=1)
    add("How many distinct orders contain at least one item from the Desserts category?",
        orders(df[df.category == "Desserts"]))
    add("What is the total revenue of Cash-paid Takeaway rows, rounded to 0 decimals?",
        df[(df.payment == "Cash") & (df.channel == "Takeaway")].revenue.sum())

    # --- 40-50 multi-step
    m = rev_by("month")
    add("What is the percentage change in total revenue from month 1 to month 2, rounded to 1 decimal?",
        (m[2] - m[1]) / m[1] * 100, d=1)
    cat = df.groupby("category")[["profit", "revenue"]].sum()
    add("Which category has the highest profit margin (profit / revenue)?", rank(cat.profit / cat.revenue), "str")
    aov = df.groupby("channel").revenue.sum() / o_by("channel")
    add("Which channel has the highest average order value (revenue / distinct orders)?", rank(aov), "str")
    oc = o_by("channel")
    add("What is the ratio of distinct Dine-in orders to distinct Delivery orders, rounded to 2 decimals?",
        oc["Dine-in"] / oc["Delivery"], d=2)
    add("What is the average number of rows (order lines) per distinct order, rounded to 2 decimals?",
        len(df) / n_orders, d=2)
    add("Which category has the second-highest total revenue?", rank(rev_by("category"), n=2), "str")
    add("Which item_name ranks 3rd by total units sold (sum of qty)?",
        rank(df.groupby("item_name").qty.sum(), n=3), "str")
    inc = m.diff().dropna()
    add("Which month number has the largest increase in total revenue versus the previous month?", rank(inc))
    ir = df.groupby("item_name").revenue.sum().sort_values(ascending=False)
    add("What percentage of total revenue comes from the 5 highest-revenue items, rounded to 1 decimal?",
        ir.head(5).sum() / R.sum() * 100, d=1)
    add("What is the median order value (total revenue per order_id), rounded to 2 decimals?",
        per_order.median(), d=2)
    add("How many items (item_name) have total revenue above the mean of per-item total revenue?",
        (ir > ir.mean()).sum())
    assert len(Q) == 50, len(Q)
    return Q


# ---------------------------------------------------------------- grading
def parse_answer(text):
    hits = re.findall(r"ANSWER\s*:\s*(.+)", text or "", flags=re.I)
    return hits[-1].strip() if hits else None


def grade(q, text):
    a = parse_answer(text)
    if a is None:
        return False
    if q["kind"] == "str":
        clean = lambda s: re.sub(r"[`*_\"']", "", s).strip().rstrip(".").lower()
        return clean(a) == clean(str(q["truth"]))
    a = re.sub(r"[,₹$%*`]|INR|Rs\.?", "", a, flags=re.I)
    m = re.search(r"-?\d+(?:\.\d+)?", a)
    if not m:
        return False
    return abs(float(m.group()) - float(q["truth"])) <= 0.5 * 10 ** (-q["d"]) + 1e-9


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- the agent (mirrors app.py)
def run_tool(name, args, df):
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
            return (buf.getvalue() or "ok")[:4000]
    except Exception as e:
        return f"ERROR: {e}"


def run_agent(client, model, question, df):
    system = ("You are a data analyst agent. Table `data` / DataFrame `df` columns:\n"
              f"{df.dtypes.to_string()}\n\nSample:\n{df.head(5).to_string()}\n\n"
              "Use tools to answer. Verify numbers before stating them. Be concise. "
              "Make a chart when it helps.")
    msgs = [{"role": "user", "content": question + SUFFIX}]
    steps = tool_errors = 0
    for _ in range(MAX_STEPS):
        r = client.messages.create(model=model, max_tokens=2000, system=system, tools=TOOLS, messages=msgs)
        steps += 1
        msgs.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            return "".join(b.text for b in r.content if b.type == "text"), steps, tool_errors, False
        results = []
        for b in r.content:
            if b.type == "tool_use":
                out = run_tool(b.name, b.input, df)
                tool_errors += str(out).startswith("ERROR:")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        msgs.append({"role": "user", "content": results})
    return "Hit step limit.", steps, tool_errors, True


def fake_agent(mode, q):
    t = q["truth"]
    if q["kind"] == "str":
        return f"ANSWER: {t}" if mode != "wrong" else "ANSWER: nothing"
    v = round(float(t), q["d"])
    if mode == "wrong":
        return f"ANSWER: {v + 7}"
    if mode == "messy":  # formats a real model might produce
        return f"**Result.** ANSWER: ₹{v:,.{q['d']}f}" if q["d"] != 1 else f"ANSWER: {v}%"
    return f"ANSWER: {v:.{q['d']}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="../menuiq/data/raw")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", choices=["oracle", "messy", "wrong"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="eval_results.csv")
    a = ap.parse_args()

    df = build_table(a.raw_dir)
    Q = build_questions(df)[: a.limit]
    print(f"table: {df.shape[0]:,} rows x {df.shape[1]} cols | questions: {len(Q)} | "
          f"mode: {a.dry_run or a.model}")
    client = None
    if not a.dry_run:
        from anthropic import Anthropic
        client = Anthropic()  # reads ANTHROPIC_API_KEY

    rows, errors = [], 0
    for q in Q:
        steps = terr = 0
        limit_hit = False
        try:
            if a.dry_run:
                text = fake_agent(a.dry_run, q)
            else:
                text, steps, terr, limit_hit = run_agent(client, a.model, q["q"], df)
        except Exception as e:  # API/infra failure: reported separately, counted as not passed
            text, errors = f"ERROR: {e}", errors + 1
        ok = grade(q, text)
        rows.append(dict(id=q["id"], passed=int(ok), truth=q["truth"], model_answer=parse_answer(text),
                         steps=steps, tool_errors=terr, step_limit=int(limit_hit), question=q["q"]))
        print(f"{q['id']:>2} {'PASS' if ok else 'FAIL'}  truth={q['truth']!s:.20}  got={parse_answer(text)!s:.20}")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    k, n = sum(r["passed"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    print(f"\nSCORE: {k}/{n} = {100 * k / n:.0f}%   (95% interval {100 * lo:.0f}-{100 * hi:.0f}%)")
    if not a.dry_run:
        print(f"avg steps/question: {sum(r['steps'] for r in rows) / n:.1f} | tool errors: "
              f"{sum(r['tool_errors'] for r in rows)} | step-limit hits: {sum(r['step_limit'] for r in rows)} "
              f"| API failures: {errors}")
        if errors:
            print("Some questions failed from API errors, not wrong answers: re-run before quoting the score.")
    print(f"details -> {a.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
