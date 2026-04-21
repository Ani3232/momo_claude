# ======================================================================|
# persistent_shell_tool.py                                             |
# Drop-in companion to python_repl — runs real shell commands in a     |
# persistent bash process sandboxed to the workspace directory.        |
# ======================================================================|

import subprocess
import sys
import json
import threading
from pathlib import Path
from langchain_core.tools import Tool
from rich.console import Console
from setup import workspace
console = Console()


_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"




# ======================================================================|
# PERSISTENT BASH SHELL                                                 |
# ======================================================================|
# One bash process lives for the entire conversation. Every call sends  |
# a command and reads back a JSON envelope with stdout/stderr/exit_code.|
# The shell is transparently restarted if it ever dies.                 |

_bash_process = None

# Tiny wrapper script that runs as the driver process.
# Workspace is injected via the AGENT_WORKSPACE env var — no .format()
# needed, so there is zero risk of brace-escaping corruption.
_BASH_DRIVER = r"""
import sys, json, subprocess, os

WORKSPACE = os.path.abspath(os.environ["AGENT_WORKSPACE"])

while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line.strip())
    if msg.get("type") == "exit":
        break

    cmd = msg["cmd"]
    env = os.environ.copy()
    env["HOME"] = WORKSPACE

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        env=env,
    )

    print(json.dumps({
        "stdout":    proc.stdout,
        "stderr":    proc.stderr,
        "exit_code": proc.returncode,
    }), flush=True)
"""


def _get_bash() -> subprocess.Popen:
    """Return the singleton bash driver process, restarting if dead."""
    global _bash_process
    if _bash_process is None or _bash_process.poll() is not None:
        import os
        env = os.environ.copy()
        env["AGENT_WORKSPACE"] = workspace   # driver reads this — no brace escaping needed
        _bash_process = subprocess.Popen(
            [sys.executable, "-c", _BASH_DRIVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workspace,
            text=True,
            bufsize=1,
            env=env,
        )
    return _bash_process


def run_persistent_bash(command: str, timeout: int = 60) -> str:
    """
    Execute *command* in the persistent bash driver.

    Each call runs in its own subprocess.run() inside the driver, so
    there is NO shared shell state between calls (no cd, no exported
    vars). If you need stateful multi-step work use the python_repl
    tool instead.

    Returns a human-readable string with stdout, stderr (if any), and
    a non-zero exit code note (if any).
    """
    shell = _get_bash()

    try:
        shell.stdin.write(json.dumps({"type": "exec", "cmd": command}) + "\n")
        shell.stdin.flush()
    except BrokenPipeError:
        global _bash_process
        _bash_process = None
        return "Error: shell pipe broke — restarted. Please retry."

    result_line: str | None = None

    def _read():
        nonlocal result_line
        result_line = shell.stdout.readline()

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        shell.kill()
        _bash_process = None
        return f"Error: command timed out after {timeout}s — shell restarted."

    if not result_line:
        _bash_process = None
        return "Error: shell process died unexpectedly."

    try:
        r = json.loads(result_line)
    except json.JSONDecodeError:
        return f"Error: could not parse shell response: {result_line!r}"

    parts: list[str] = []
    
    console.print(_BULLET," [bold cyan]$ Bash Tool[/bold cyan]")
    console.print(_NEST,f"[dim green]{command}[/dim green]")
    console.print("-"*80,"||")
    if r["stdout"]:
        parts.append(r["stdout"].rstrip())
        console.print("\n[bold green]Stdout:[/bold green]", end="")
        console.print(r["stdout"].rstrip())

    if r["stderr"]:
        parts.append(f"[stderr]\n{r['stderr'].rstrip()}")
        console.print("\n[bold red]Stderr:[/bold red]",end="")
        console.print(f"[stderr]{r['stderr'].rstrip()}")

    if r["exit_code"] != 0:
        parts.append(f"[exit code: {r['exit_code']}]")
        console.print("\n[yellow]EXIT CODE: [/yellow]", end="")
        console.print(f"[exit code: {r['exit_code']}]")
    console.print("-"*80,"||")
    print("")
        

    return "\n".join(parts) if parts else "[command completed with no output]"


# ======================================================================|
# LANGCHAIN TOOL DEFINITION                                             |
# ======================================================================|

bash_tool = Tool(
    name="bash_shell",
    func=run_persistent_bash,
    description=(
        "Runs a shell command on the host system inside the workspace directory. "
        "Use for filesystem operations (ls, mkdir, cat, cp, mv, rm), "
        "running scripts (python script.py, node index.js), "
        "installing packages (pip install ..., npm install ...), "
        "or any other OS-level command. "
        "Each call is independent — working directory always resets to workspace. "
        "For stateful multi-step Python work, prefer python_repl instead. "
        "Input: a single shell command string, e.g. 'ls -la' or 'pip install requests'."
    ),
)