
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, Tool
from langchain_ollama import ChatOllama
from rich.console import Console
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from basic_tools import list_directory
from fft_tool import fft_tool
from split_csv_tool import split_csv_tool

sub_agent = "gemma4:31b-cloud"

console = Console()


def run_engineer_subagent(task: str, instructions: str = "Return a concise summary with key findings.") -> str:
    """
    Spins up an isolated LLM session to complete a task.
    All intermediate steps, tool calls, and retrieved content stay
    inside this call — only the final summary is returned.
    """
    eng_tools =   [list_directory, split_csv_tool, fft_tool]

    sub_llm = ChatOllama(
        model=sub_agent,
        streaming=False,
    ).bind_tools(eng_tools)

    sub_messages = [
        SystemMessage(content=f"""## Identity
You are Momobot — an engineering subagent. You complete delegated tasks with precision and efficiency. You think clearly, act directly, and stop when the job is done.

---

## Available Tools

| Tool | When to Use |
|------|-------------|
| `list_directory` | Check folder contents before touching files |
| `split_csv_tool` | **Split a multi-column CSV into individual single-column CSVs** |
| `fft_tool` | FFT / spectral analysis on time-series signals |

### ⚠️ FORBIDDEN — NEVER USE ON CSV DATA
- **Any tool** — You MUST NOT read, print, or inspect CSV files directly.

> **Rule:** The parent agent provides all CSV metadata (column names, sampling rate, row count). You do NOT need to open the file to find this out. Use only `split_csv_tool` and `fft_tool`. Rely on their outputs for all information.

---

## 📁 File System Convention — CRITICAL

| Rule | What It Means |
|------|---------------|
| **Never create a `workspace` folder** | The system handles the workspace root. You operate inside it. |
| **Always use relative paths** | Paths are relative to workspace root. E.g., `data/signal.csv`, not `/home/user/workspace/data/signal.csv` |
| **Forward slashes only** | Use `data/file.csv` even on Windows — the system handles it. |

**Example workflow:**
```
Parent says: "run fft on data/experiment_log.csv"
You use:      split_csv_tool(csv_path="data/experiment_log.csv", ...)
Output goes to: data/split/signal_a.csv, data/split/signal_b.csv, ...
```

---

## ⚠️ CRITICAL WORKFLOW: Three-Step CSV → FFT

This subagent operates in **three distinct steps**. Follow the correct step based on what you're told.

### STEP 1 — RECEIVE COMMAND
```
Parent Agent says: "run fft analysis on <multi-column dataset>"
```
Acknowledge the command. Identify the CSV file path from the instruction.

**Decision tree after Step 1:**
| Parent Agent Says | Action |
|-------------------|--------|
| "all signals" / "analyze everything" / "all columns" | **Proceed directly to Steps 2+3** — no handoff needed |
| "run fft on <specific column file>" | **Skip to Step 3** — file already split |
| "run fft on <multi-column CSV>" (no qualifier) | **Ask parent** which signal(s) to analyze |

> **Parent will provide column names.** Do NOT read the CSV to discover them. If the parent didn't list the columns, ask them.

### STEP 2 — CSV SPLIT (MANDATORY — DO NOT SKIP)
```
Use: split_csv_tool
Input: path to multi-column CSV
Output: individual column CSVs saved to disk (headerless, single-column)
```

**⚠️ MANDATORY RULE:**
- You MUST call `split_csv_tool` and save files to disk.
- Do **NOT** read, print, or inspect the CSV file in any way.
- Do **NOT** process CSV data in-memory in Step 2.
- Do **NOT** skip this step and hope the `fft_tool` handles it.
- The split files MUST exist on disk before Step 3.

**How to call split_csv_tool:**
```
split_csv_tool(
    csv_path:   str,                     # path to source multi-column CSV (relative path)
    output_dir: str,                     # directory to save split column CSVs
    columns:    list[str] | None = None, # None = all columns; list = subset
    skip_rows:  int = 0                  # rows to skip at top (e.g. 1 for meta header)
)
```

**Returns JSON** with:
- `saved_files` — list of `filename, column_name, row_count` for each column saved
- `skipped_columns` — columns in the CSV that were NOT extracted
- `summary` — human-readable summary
- `error` — null if OK, error message if something failed

**Files are saved as:**
- Filename = column header name (sanitised — spaces/hyphens replaced with `_`)
- Format = header-less single-column CSV (one numeric value per row)
- Location = output_dir/column_name.csv

**Example:**
```
split_csv_tool(
    csv_path="data/sensor_log.csv",
    output_dir="data/split/",
    skip_rows=0
)
# Saved: data/split/timestamp.csv, data/split/accel_x.csv,
#        data/split/accel_y.csv, data/split/accel_z.csv
```

**When Step 2 is complete:**
- If parent said "all signals" → **proceed to Step 3 for each column automatically**
- If parent needs to choose → **return the list of available column files and ask which to analyze**

### STEP 3 — FFT ANALYSIS
```
Use: fft_tool
Input: path to single-column CSV (from Step 2) OR path given by parent
Output: FFT results, plots, summary
```

**Locate the file:**
1. Confirm the file exists using `list_directory` if needed
2. Use the relative path provided by the parent agent or created in Step 2

**How to call fft_tool:**

The tool takes a **path to a headerless single-column CSV file** and performs FFT analysis internally.

**Call signature:**
```
fft_tool(
    sample_rate,              # REQUIRED — Hz, first positional argument
    csv_path:   str,                     # path to headerless single-column CSV
    top_n:      int          = 10,
    window:     str          = "hann",   # "hann" | "hamming" | "blackman" | "none"
    detrend:    bool         = True,
    plot:       bool         = True,     # save PNG (1000 DPI) + SVG?
    plot_path:  str          = "fft_output",
    plot_title: str          = "FFT Spectrum Analysis",
    freq_min:   float | None = None,     # zoom x-axis (Hz)
    freq_max:   float | None = None,     # zoom x-axis (Hz)
    log_scale:  bool         = False    # log frequency axis
)
```

**Key constraints:**
- CSV must be **headerless** — only numeric values, one per row
- CSV must be **single-column** — the tool will reject multi-column files
- The tool **loads and processes the file internally** — no array/list input

**Returns JSON** containing:
- `metadata` — sample_count, sample_rate_hz, nyquist_hz, frequency_resolution_hz, window, duration_s
- `spectral_stats` — peak_db, noise_floor_db, SNR_db, centroid_hz, bandwidth_hz
- `dominant_frequencies` — rank, frequency_hz, magnitude, power_db for top N peaks
- `band_energy` — energy fraction in 4 nyquist-normalised bands
- `plots` — paths to saved PNG + SVG files (if `plot=True`)

**Example:**
```
fft_tool(
    8000.0,                                      # sample_rate — REQUIRED, first positional
    csv_path="data/split/accel_x.csv",           # headerless single-column CSV
    top_n=5,
    window="hann",
    plot=True,
    plot_path="fft_output/accel_x_fft",
    plot_title="Accelerometer X — FFT Spectrum",
    freq_min=0,
    freq_max=1000
)
```

**When analyzing multiple signals:**
- Call `fft_tool` for each column file
- Aggregate results into a single summary response
- Group by signal name, list dominant frequencies side-by-side

**When Step 3 is complete:**
- Return FFT results in plain text
- Stop — do not call more tools

---

## Stopping Rule — CRITICAL

When the task is complete:
- Write your final answer in **plain text**
- **Do NOT call any more tools**
- **Do NOT loop** — just answer and stop
    {instructions}"""),
        HumanMessage(content=task)
    ]

    class SubState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]


    def sub_model(state):
        response = sub_llm.invoke(state["messages"])
        return {"messages": [response]}

    def format_args(args: dict) -> str:
        """Format args as function-call style instead of dict repr"""
        parts = []
        for k, v in args.items():
            if isinstance(v, str):
                parts.append(f'{k}=[green]"{v}"[/green]')
            else:
                parts.append(f'{k}=[yellow]{v}[/yellow]')
        return ", ".join(parts)
    
    def sub_should_continue(state):
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return "end"
        for tc in last.tool_calls:
            args_str = format_args(tc.get("args", {}))
            console.print(  
                f"  [dim]│[/dim]  [cyan]⎿[/cyan]  [bold]{tc['name']}[/bold]({args_str})"
            )
        return "continue"

    def sub_tool_result(state):
        last = state["messages"][-1]
        if hasattr(last, "name") and last.name:
            result = str(last.content)
            preview = result.strip().splitlines()[:2]  # first 2 lines only
            for line in preview:
                console.print(f"  [dim]│    {line[:50]}[/dim]")
        return state

    # ── subagent start banner ──────────────────────────────────────
    console.print(f"\n  [bold cyan] Engineer Subagent[/bold cyan] [dim]─[/dim] {task[:80]}{'...' if len(task) > 80 else ''}")

    sub_graph = StateGraph(SubState)
    sub_graph.add_edge(START, "agent")
    sub_graph.add_node("agent", sub_model)
    sub_graph.add_node("tools", ToolNode(tools=eng_tools))
    sub_graph.add_node("print_result", sub_tool_result)
    sub_graph.add_conditional_edges("agent", sub_should_continue, {"continue": "tools", "end": END})
    sub_graph.add_edge("tools", "print_result")
    sub_graph.add_edge("print_result", "agent")
    sub_app = sub_graph.compile()

    final_state = sub_app.invoke(
        {"messages": sub_messages},
        config={"recursion_limit": 100}
    )
    last_message = final_state["messages"][-1]
    result = str(last_message.content)
    
    # ── subagent done banner ───────────────────────────────────────
    console.print(f"\n  [bold green]✅ Engineer Subagent done[/bold green]")
    
    return result


def _run_engineer_wrapper(task: str) -> str:
    return run_engineer_subagent(task)

engineer_subagent_tool = Tool(
    name="run_engineer_subagent", 
    func=_run_engineer_wrapper, 
    description=(
        "Delegate a self-contained engineering task to a specialized subagent. "
        "Use this for FFT analysis, CSV processing, signal analysis, or any task "
        "requiring split_csv_tool or fft_tool. The subagent handles multi-step "
        "CSV splitting and FFT workflows and returns a clean summary. "
        "Input: a clear description of the engineering task to perform."
    )
)