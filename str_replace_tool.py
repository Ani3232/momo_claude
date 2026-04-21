from langchain_core.tools import Tool
import difflib
from pathlib import Path
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from setup import workspace
import re



_console = Console()
_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"





def str_replace_file(input_data) -> str:
    """
    Edit a specific string inside a file without rewriting the whole thing.
    
    Input format (JSON string or dict):
    {
        "path": "relative/path/to/file.py",
        "old_str": "exact string to find and replace",
        "new_str": "replacement string"
    }
    
    Rules:
    - old_str must appear EXACTLY once in the file.
    - Paths are relative to the workspace directory.
    - Returns a unified diff showing what changed.
    """
    import json

    if isinstance(input_data, dict):
        args = input_data
    else:
        try:
            args = json.loads(input_data)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON input — {e}"

    path    = args.get("path", "").strip()
    old_str = args.get("old_str", "")
    new_str = args.get("new_str", "")

    if not path:
        return "Error: 'path' is required."
    if old_str == "":
        return "Error: 'old_str' cannot be empty."

    full_path = Path(workspace) / path

    if not full_path.exists():
        return f"Error: File not found — {path}"
    if not full_path.is_file():
        return f"Error: '{path}' is not a file."

    original = full_path.read_text(encoding="utf-8")

    count = len(re.findall(re.escape(old_str), original))
    if count == 0:
        return (
            f"Error: 'old_str' not found in {path}.\n"
            "Tip: Make sure whitespace and indentation match exactly."
        )
    if count > 1:
        return (
            f"Error: 'old_str' appears {count} times in {path}. "
            "It must be unique. Add more context to make it unambiguous."
        )

    updated = original.replace(old_str, new_str, 1)
    full_path.write_text(updated, encoding="utf-8")

    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))

    if not diff_lines:
        return "No changes made (old_str and new_str were identical)."


    

    
    _console.print(f"{_BULLET} [bold]Editing File[/bold]")
    _console.print(f"{_NEST} [dim] {path}[/dim]")
    
    for line in diff_lines:
        if line.startswith("---") or line.startswith("+++"):
            continue
        elif line.startswith("@@"):
            _console.print(f"       [dim]{escape(line)}[/dim]",end="")
            _console.print("")
        elif line.startswith("-"):
            _console.print(f"       [red]{escape(line)}[/red]",end="")
            _console.print("")
        elif line.startswith("+"):
            _console.print(f"       [green]{escape(line)}[/green]",end="")
            _console.print("")
        else:
            _console.print("        ",escape(line),end="")
            _console.print("")

    _console.print("")
    # ─────────────────────────────────────────────────────────────────────

    diff_output = "".join(diff_lines)
    return f"✓ Edit applied to `{path}`\n\n```diff\n{diff_output}\n```"


str_replace_tool = Tool(
    name="str_replace_file",
    func=str_replace_file,
    description=(
        "Edit a specific string in a file without rewriting the whole file. "
        "Use this after reading a file when you want to change a specific section. "
        "Input must be a JSON string with keys: "
        "'path' (relative), "
        "'old_str' (exact text to find — must appear exactly once), "
        "'new_str' (replacement text). "
        "Returns a diff showing lines removed (-) and added (+)."
    ),
)


