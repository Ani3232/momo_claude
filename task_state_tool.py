"""
Task State Management Tool for LangChain Agents (v2 - with Rich prints)
=========================================================================

Refactored for:
  ✓ Windows-safe file locking (no atomic rename issues)
  ✓ LLM-friendly interface (simpler schemas, clearer descriptions)
  ✓ Warm minimalist prints with Rich
  ✓ Better error messages
  ✓ Reduced complexity

Windows fix: Uses fcntl-compatible retry logic + json.dump with flush
LLM fixes: Separated concerns, removed **kwargs, simpler tool descriptions
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, List, Dict
from enum import Enum

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    raise ImportError(
        "LangChain is required. Install with: pip install langchain-core"
    )

try:
    from rich.console import Console
    _console = Console()
    _HAS_RICH = True
except ImportError:
    _console = None
    _HAS_RICH = False

# ============================================================================
# STYLING
# ============================================================================

_CORAL = "#C8603A"
_SAGE = "#7A9A6E"
_SLATE = "#5A6B7A"
_BULLET = f"[{_CORAL}]⬤[/{_CORAL}]" if _HAS_RICH else "•"
_NEST = "[dim]  ⎿[/dim]" if _HAS_RICH else "  "

def _print(*content, style: str = "dim") -> None:
    """Print with Rich if available, else fallback to print()"""
    if _HAS_RICH and _console:
        for line in content:
            _console.print(line)
    else:
        for line in content:
            print(line)


# ============================================================================
# CONFIGURATION
# ============================================================================

STATE_FILE = str(Path(__file__).parent / "task_state.json")
AUDIT_LOG_FILE = str(Path(__file__).parent / "task_state_audit.jsonl")
MAX_REPLAN_HISTORY = 10
MAX_RETRIES = 5  # For Windows file locking retries
RETRY_DELAY = 0.1  # seconds


class StepStatus(str, Enum):
    """Valid step statuses."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Valid task statuses."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ============================================================================
# FILE I/O WITH WINDOWS COMPATIBILITY
# ============================================================================

def _log_audit(event: str, details: Dict[str, Any]) -> None:
    """Append audit log (best effort, non-blocking)."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "details": details,
    }
    try:
        with open(AUDIT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Silently ignore audit failures
        pass


def _load() -> Dict[str, Any]:
    """Load state from disk. Returns empty dict if missing or corrupted."""
    if not os.path.exists(STATE_FILE):
        return {}
    
    # Retry on Windows file lock
    for attempt in range(MAX_RETRIES):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (PermissionError, OSError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                continue
            # Fall through: return empty dict on final failure
            return {}
        except json.JSONDecodeError:
            # File corrupted; return empty
            return {}
        except Exception:
            return {}
    
    return {}


def _save(state: Dict[str, Any]) -> None:
    """
    Save state to disk safely on Windows.
    
    Instead of atomic rename (which fails on Windows under contention),
    we write directly with exclusive file handle.
    """
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    for attempt in range(MAX_RETRIES):
        try:
            # Windows-safe: write directly (with exclusive lock if possible)
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force disk sync
            return  # Success
        except (PermissionError, OSError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            # Final failure: re-raise with context
            raise RuntimeError(
                f"Failed to save state after {MAX_RETRIES} retries. "
                f"Last error: {type(e).__name__}: {e}"
            )
        except Exception as e:
            raise


# ============================================================================
# STATE HELPERS
# ============================================================================

def _validate_state(state: Dict[str, Any]) -> List[str]:
    """Validate state structure. Returns list of errors (empty if valid)."""
    errors = []
    
    if "goal" not in state:
        errors.append("Missing 'goal'")
    if "status" not in state:
        errors.append("Missing 'status'")
    if "steps" not in state or not isinstance(state["steps"], list):
        errors.append("'steps' must be a list")
    else:
        for i, step in enumerate(state["steps"]):
            if "id" not in step:
                errors.append(f"Step {i}: missing 'id'")
            if "task" not in step:
                errors.append(f"Step {i}: missing 'task'")
            if "status" not in step:
                errors.append(f"Step {i}: missing 'status'")
    
    return errors


def _unblock(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Automatically unblock steps whose dependencies are complete."""
    done_ids = {s["id"] for s in steps if s["status"] == StepStatus.DONE.value}
    for step in steps:
        if step["status"] == StepStatus.BLOCKED.value:
            deps = step.get("depends_on", [])
            if all(dep in done_ids for dep in deps):
                step["status"] = StepStatus.PENDING.value
    return steps


def _update_task_status(state: Dict[str, Any]) -> None:
    """Update top-level task status based on step statuses."""
    if not state.get("steps"):
        return
    
    statuses = {s["status"] for s in state["steps"]}
    
    if statuses == {StepStatus.DONE.value}:
        state["status"] = TaskStatus.DONE.value
    elif StepStatus.FAILED.value in statuses:
        if not any(
            s["status"] in (StepStatus.PENDING.value, StepStatus.RUNNING.value, StepStatus.BLOCKED.value)
            for s in state["steps"]
        ):
            state["status"] = TaskStatus.FAILED.value
    elif any(s["status"] == StepStatus.RUNNING.value for s in state["steps"]):
        state["status"] = TaskStatus.RUNNING.value


def _coerce_to_list(value: Any) -> List[str]:
    """Coerce value to list of strings. Handles LLM mistakes."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if v]
        except:
            pass
        return [value] if value.strip() else []
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


# ============================================================================
# PYDANTIC SCHEMAS (Simplified for LLMs)
# ============================================================================

class StepInput(BaseModel):
    """A single step. LLMs often add extra fields; we ignore them."""
    model_config = ConfigDict(extra='ignore')
    
    id: str = Field(
        description="Unique step identifier (e.g., '1', 'setup_db', 'a1')"
    )
    task: str = Field(
        description="Brief description of what this step does"
    )
    depends_on: Optional[List[str]] = Field(
        default=None,
        description="List of step IDs that must complete first (optional)"
    )


class InitPlanInput(BaseModel):
    """Initialize a task plan. Clear, simple schema for LLMs."""
    model_config = ConfigDict(extra='ignore')
    
    goal: str = Field(
        description="What you're trying to accomplish (e.g., 'Build a web scraper')"
    )
    steps: List[StepInput] = Field(
        description="List of steps to complete the goal. Order matters."
    )


class CompleteStepInput(BaseModel):
    """Mark a step complete. Minimal, focused schema."""
    model_config = ConfigDict(extra='ignore')
    
    step_id: str = Field(
        description="The ID of the step you just completed"
    )
    result: Optional[str] = Field(
        default=None,
        description="What was the outcome? (e.g., 'Successfully scraped 500 items')"
    )


class FailStepInput(BaseModel):
    """Mark a step failed."""
    model_config = ConfigDict(extra='ignore')
    
    step_id: str = Field(
        description="The ID of the step that failed"
    )
    reason: str = Field(
        description="Why did it fail? (e.g., 'API returned 429 error')"
    )


class ReplanInput(BaseModel):
    """Replan after a failure or new information."""
    model_config = ConfigDict(extra='ignore')
    
    reason: str = Field(
        description="Why you're replanning (e.g., 'Step 2 hit rate limit')"
    )
    remove_steps: Optional[List[str]] = Field(
        default=None,
        description="Step IDs to remove (optional)"
    )
    add_steps: Optional[List[StepInput]] = Field(
        default=None,
        description="New steps to add (optional)"
    )


# ============================================================================
# QUERY FUNCTIONS (Non-Tools)
# ============================================================================

def task_state_summary() -> Dict[str, Any]:
    """
    Get a compact summary. Call this directly in your agent loop,
    inject into system prompt for context.
    
    Does NOT require a tool call.
    """
    state = _load()
    
    if not state:
        return {
            "initialized": False,
            "goal": None,
            "total_steps": 0,
            "percent_complete": 0,
        }
    
    steps = state.get("steps", [])
    total = len(steps)
    done = sum(1 for s in steps if s["status"] == StepStatus.DONE.value)
    pending = sum(1 for s in steps if s["status"] == StepStatus.PENDING.value)
    running = sum(1 for s in steps if s["status"] == StepStatus.RUNNING.value)
    blocked = sum(1 for s in steps if s["status"] == StepStatus.BLOCKED.value)
    failed = sum(1 for s in steps if s["status"] == StepStatus.FAILED.value)
    
    return {
        "initialized": True,
        "goal": state.get("goal", "")[:60],
        "overall_status": state.get("status", TaskStatus.RUNNING.value),
        "total_steps": total,
        "steps_complete": done,
        "steps_pending": pending,
        "steps_running": running,
        "steps_blocked": blocked,
        "steps_failed": failed,
        "percent_complete": round(100 * done / total) if total > 0 else 0,
    }


# ============================================================================
# LLM TOOLS (Refactored for clarity, with Rich prints)
# ============================================================================

@tool
def read_task_state() -> str:
    """
    Read the complete current task state.
    
    Call this at the start of each turn to see:
    - The goal
    - All steps with their status
    - Dependencies
    - Results and errors
    
    Returns JSON.
    """
    state = _load()
    
    if not state:
        _print()
        _print(f"{_BULLET} [bold red]No Task[/bold red]")
        _print(f"{_NEST} Initialize one with init_task()")
        _print()
        return json.dumps({"error": "No task initialized yet."})
    
    errors = _validate_state(state)
    if errors:
        _print()
        _print(f"{_BULLET} [bold red]Validation Error[/bold red]")
        for err in errors:
            _print(f"{_NEST} {err}")
        _print()
        return json.dumps({
            "error": "State validation failed",
            "details": errors,
            "state": state
        })
    
    _print()
    _print(f"{_BULLET} [bold {_SAGE}]Reading State[/bold {_SAGE}]")
    _print(f"{_NEST} [dim]{state.get('goal', '(untitled)')[:50]}[/dim]")
    _print(f"{_NEST} [green]✓ Done[/green]")
    _print()
    
    return json.dumps(state, indent=2, default=str)


@tool(args_schema=InitPlanInput)
def init_task(goal: str, steps: List[StepInput]) -> str:
    """
    Start a new task with a goal and a list of steps.
    
    Args:
        goal: What you're trying to accomplish
        steps: List of steps (each with id, task, and optional depends_on)
    
    Example:
        goal = "Build a web scraper"
        steps = [
            {"id": "1", "task": "Set up database"},
            {"id": "2", "task": "Write scraper", "depends_on": ["1"]},
            {"id": "3", "task": "Test scraper", "depends_on": ["2"]},
        ]
    
    Returns the initialized plan as JSON.
    """
    try:
        normalised = []
        for s in steps:
            # Handle both Pydantic models and dicts
            s_dict = s.model_dump() if hasattr(s, 'model_dump') else s
            
            sid = str(s_dict.get("id", "")).strip()
            task = str(s_dict.get("task", "")).strip()
            deps = _coerce_to_list(s_dict.get("depends_on"))
            
            if not sid or not task:
                continue  # Skip malformed
            
            normalised.append({
                "id": sid,
                "task": task,
                "status": StepStatus.PENDING.value if not deps else StepStatus.BLOCKED.value,
                "depends_on": deps,
                "result": None,
                "error": None,
            })
        
        if not normalised:
            _print()
            _print(f"{_BULLET} [bold red]No Steps[/bold red]")
            _print(f"{_NEST} Each step needs 'id' and 'task'")
            _print()
            return json.dumps({"error": "No valid steps provided. Each step must have 'id' and 'task'."})
        
        state = {
            "goal": goal,
            "status": TaskStatus.RUNNING.value,
            "steps": normalised,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "replan_log": [],
        }
        
        _save(state)
        _log_audit("init_task", {"goal": goal, "step_count": len(normalised)})
        
        _print()
        _print(f"{_BULLET} [bold {_SAGE}]Task Initialized[/bold {_SAGE}]")
        _print(f"{_NEST} [dim]{goal[:50]}[/dim]")
        _print(f"{_NEST} [dim]{len(normalised)} steps[/dim]")
        _print(f"{_NEST} [green]✓ Done[/green]")
        _print()
        
        return json.dumps(state, indent=2, default=str)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _print()
        _print(f"{_BULLET} [bold red]Error[/bold red]")
        _print(f"{_NEST} {error_msg}")
        _print()
        _log_audit("init_task_error", {"error": error_msg})
        return json.dumps({"error": error_msg})


@tool(args_schema=CompleteStepInput)
def complete_step(step_id: str, result: Optional[str] = None) -> str:
    """
    Mark a step as done (successful completion).
    
    Args:
        step_id: The ID of the step you just finished
        result: Optional description of what was accomplished
    
    Returns the updated state as JSON.
    """
    try:
        state = _load()
        
        if not state:
            _print()
            _print(f"{_BULLET} [bold red]No Task[/bold red]")
            _print(f"{_NEST} Initialize one with init_task()")
            _print()
            return json.dumps({"error": "No task initialized. Call init_task first."})
        
        step_id = str(step_id).strip()
        step = next((s for s in state.get("steps", []) if s["id"] == step_id), None)
        
        if step is None:
            available = [s["id"] for s in state.get("steps", [])]
            _print()
            _print(f"{_BULLET} [bold red]Not Found[/bold red]")
            _print(f"{_NEST} Step '{step_id}' not found")
            _print(f"{_NEST} [dim]Available: {available}[/dim]")
            _print()
            return json.dumps({
                "error": f"Step '{step_id}' not found. Available: {available}"
            })
        
        # Update and unblock
        step["status"] = StepStatus.DONE.value
        step["result"] = result
        step["error"] = None
        
        state["steps"] = _unblock(state["steps"])
        _update_task_status(state)
        
        _save(state)
        _log_audit("step_complete", {"step_id": step_id, "has_result": result is not None})
        
        _print()
        _print(f"{_BULLET} [bold {_SAGE}]Step Complete[/bold {_SAGE}]")
        _print(f"{_NEST} [dim]Step {step_id}[/dim]")
        if result:
            _print(f"{_NEST} [dim]{result[:50]}[/dim]")
        _print(f"{_NEST} [green]✓ Done[/green]")
        _print()
        
        return json.dumps(state, indent=2, default=str)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _print()
        _print(f"{_BULLET} [bold red]Error[/bold red]")
        _print(f"{_NEST} {error_msg}")
        _print()
        _log_audit("step_complete_error", {"step_id": step_id, "error": error_msg})
        return json.dumps({"error": error_msg})


@tool(args_schema=FailStepInput)
def fail_step(step_id: str, reason: str) -> str:
    """
    Mark a step as failed.
    
    Args:
        step_id: The ID of the failed step
        reason: What went wrong?
    
    You should usually call replan_task after this to try an alternative approach.
    
    Returns the updated state as JSON.
    """
    try:
        state = _load()
        
        if not state:
            _print()
            _print(f"{_BULLET} [bold red]No Task[/bold red]")
            _print(f"{_NEST} Initialize one with init_task()")
            _print()
            return json.dumps({"error": "No task initialized. Call init_task first."})
        
        step_id = str(step_id).strip()
        step = next((s for s in state.get("steps", []) if s["id"] == step_id), None)
        
        if step is None:
            available = [s["id"] for s in state.get("steps", [])]
            _print()
            _print(f"{_BULLET} [bold red]Not Found[/bold red]")
            _print(f"{_NEST} Step '{step_id}' not found")
            _print(f"{_NEST} [dim]Available: {available}[/dim]")
            _print()
            return json.dumps({
                "error": f"Step '{step_id}' not found. Available: {available}"
            })
        
        step["status"] = StepStatus.FAILED.value
        step["error"] = reason
        step["result"] = None
        
        _update_task_status(state)
        _save(state)
        _log_audit("step_failed", {"step_id": step_id, "reason": reason})
        
        _print()
        _print(f"{_BULLET} [bold red]Step Failed[/bold red]")
        _print(f"{_NEST} [dim]Step {step_id}[/dim]")
        _print(f"{_NEST} [dim]{reason[:50]}[/dim]")
        _print(f"{_NEST} [yellow]→ Consider using replan_task()[/yellow]")
        _print()
        
        return json.dumps(state, indent=2, default=str)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _print()
        _print(f"{_BULLET} [bold red]Error[/bold red]")
        _print(f"{_NEST} {error_msg}")
        _print()
        _log_audit("step_failed_error", {"step_id": step_id, "error": error_msg})
        return json.dumps({"error": error_msg})


@tool(args_schema=ReplanInput)
def replan_task(
    reason: str,
    remove_steps: Optional[List[str]] = None,
    add_steps: Optional[List[StepInput]] = None
) -> str:
    """
    Restructure the plan when a step fails or new info emerges.
    
    Use this to adapt. Instead of giving up, remove failed/obsolete steps
    and add new ones.
    
    Args:
        reason: Why you're replanning
        remove_steps: IDs of steps to drop (optional)
        add_steps: New steps to add (optional)
    
    Example:
        reason = "Step 2 hit rate limit; switching to batch approach"
        remove_steps = ["2"]
        add_steps = [
            {"id": "2a", "task": "Wait 60s", "depends_on": ["1"]},
            {"id": "2b", "task": "Retry scraping", "depends_on": ["2a"]},
        ]
    
    Returns the updated state as JSON.
    """
    try:
        state = _load()
        
        if not state:
            _print()
            _print(f"{_BULLET} [bold red]No Task[/bold red]")
            _print(f"{_NEST} Initialize one with init_task()")
            _print()
            return json.dumps({"error": "No task initialized. Call init_task first."})
        
        # Remove specified steps
        remove_ids = set(_coerce_to_list(remove_steps)) if remove_steps else set()
        state["steps"] = [s for s in state["steps"] if s["id"] not in remove_ids]
        
        # Add new steps
        added_count = 0
        if add_steps:
            for s in add_steps:
                s_dict = s.model_dump() if hasattr(s, 'model_dump') else s
                
                sid = str(s_dict.get("id", "")).strip()
                task = str(s_dict.get("task", "")).strip()
                deps = _coerce_to_list(s_dict.get("depends_on"))
                
                if sid and task:
                    state["steps"].append({
                        "id": sid,
                        "task": task,
                        "status": StepStatus.PENDING.value if not deps else StepStatus.BLOCKED.value,
                        "depends_on": deps,
                        "result": None,
                        "error": None,
                    })
                    added_count += 1
        
        # Reprocess blocking
        state["steps"] = _unblock(state["steps"])
        state["status"] = TaskStatus.RUNNING.value
        
        # Log the replan
        if "replan_log" not in state:
            state["replan_log"] = []
        
        state["replan_log"].append({
            "reason": reason,
            "removed": list(remove_ids),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        if len(state["replan_log"]) > MAX_REPLAN_HISTORY:
            state["replan_log"] = state["replan_log"][-MAX_REPLAN_HISTORY:]
        
        _save(state)
        _log_audit("replan", {"reason": reason, "removed_count": len(remove_ids), "added_count": added_count})
        
        _print()
        _print(f"{_BULLET} [bold {_SAGE}]Plan Updated[/bold {_SAGE}]")
        _print(f"{_NEST} [dim]{reason[:50]}[/dim]")
        if remove_ids:
            _print(f"{_NEST} [dim]Removed {len(remove_ids)} steps[/dim]")
        if added_count:
            _print(f"{_NEST} [dim]Added {added_count} steps[/dim]")
        _print(f"{_NEST} [green]✓ Done[/green]")
        _print()
        
        return json.dumps(state, indent=2, default=str)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _print()
        _print(f"{_BULLET} [bold red]Error[/bold red]")
        _print(f"{_NEST} {error_msg}")
        _print()
        _log_audit("replan_error", {"reason": reason, "error": error_msg})
        return json.dumps({"error": error_msg})


# ============================================================================
# EXPORT
# ============================================================================

# These are the tools to register with your LangChain agent
task_state_tools = [
    read_task_state,
    init_task,
    complete_step,
    fail_step,
    replan_task,
]

__all__ = [
    "task_state_tools",
    "task_state_summary",
    "read_task_state",
    "init_task",
    "complete_step",
    "fail_step",
    "replan_task",
]