# ======================================================================||
# ORCHESTRATOR AS SUBAGENT TOOL
# ======================================================================||
# This transforms the main agent orchestrator into a reusable tool
# that can be delegated to by other agents.
#
# Key design:
# - Isolated LLM session (ChatOllama for gemma4)
# - Full memory cascade (working, episodic, semantic)
# - Tool binding and execution
# - Returns clean summary to parent agent
# ======================================================================||

import operator
import sys
import time
from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict, Union, List, Optional, Sequence
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool, Tool, tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from rich.console import Console
import asyncio
from llama_cloud import AsyncLlamaCloud
from setup import workspace
from basic_tools import base_tools
from clarification_tool import ask_clarifying_questions_tool
from task_state_tool import task_state_tools
from str_replace_tool import str_replace_tool
console = Console()

# ======================================================================
# CONFIGURATION
# ======================================================================
_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"

ORCHESTRATOR_MODEL = "gemma4:31b-cloud"
ORCHESTRATOR_CTX_WINDOW = 256000
ORCHESTRATOR_CHAR_DELAY = 0.01  # For streaming output (if verbose mode)




load_dotenv()
API = os.getenv("Parser_API")
# ======================================================================||
#  Parsing Tool
# ======================================================================||
_llama_client = AsyncLlamaCloud(api_key=API)


async def _parse_pdf_async(file_path: str, tier: str = "agentic") -> str:
    file_obj = await _llama_client.files.create(file=file_path, purpose="parse")
    result = await _llama_client.parsing.parse(
        file_id=file_obj.id,
        tier=tier,
        version="latest",
        expand=["markdown_full", "text_full"],
    )
    return result.markdown_full or result.text_full or "No content extracted"


@tool
def parse_pdf(file_path: str, tier: str = "agentic") -> str:
    """Parse a PDF and return its content as markdown.
    file_path is relative to workspace (e.g. 'report.pdf' or 'papers/doc.pdf')
    tier options: fast, cost_effective, agentic, agentic_plus
    """
    full_path = Path(workspace) / file_path
    if not full_path.exists():
        # Also try absolute path in case agent passes full path
        full_path = Path(file_path)
    if not full_path.exists():
        return f"❌ File not found: {file_path}. Use list_directory('.') to check available files."
    try:
        # Always spin a NEW event loop — safe in any thread/sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_parse_pdf_async(str(full_path), tier))
        finally:
            loop.close()
    except Exception as e:
        return f"❌ PDF parsing failed: {str(e)}"



# ======================================================================
# MEMORY CASCADE (from original agent)
# ======================================================================

class MemoryCascade:
    """
    Manages the three-tier memory system for the orchestrator.
    - Injects relevant context into LLM prompts
    - Compacts episodic → semantic on overflow
    - Maintains budget constraints
    """
    
    EPISODIC_CAPACITY = 10
    SEMANTIC_MAX_TOKENS = 500
    BASE_SYSTEM_PROMPT = None
    
    @staticmethod
    def set_base_prompt(prompt: str):
        """Set the base system prompt."""
        MemoryCascade.BASE_SYSTEM_PROMPT = prompt
    
    @staticmethod
    def build_system_prompt(state: 'OrchestratorState', base_prompt: str) -> str:
        """
        Construct the system prompt with memory injection.
        """
        # Use provided base prompt
        base = base_prompt
        
        # Build memory context
        episodic_str = ""
        if state["episodic_memory"]:
            episodic_str = "Recent observations:\n" + "\n".join(
                f"  • {obs}" for obs in state["episodic_memory"][-5:]
            )
        
        semantic_str = ""
        if state["semantic_memory"]:
            semantic_str = f"Project context:\n{state['semantic_memory']}"
        
        memory_context = ""
        if semantic_str:
            memory_context += semantic_str + "\n\n"
        if episodic_str:
            memory_context += episodic_str + "\n"
        
        system = f"""{base}

{memory_context}"""
        
        return system
    
    @staticmethod
    def add_observation(state: 'OrchestratorState', observation: str) -> dict:
        """Add observation to episodic memory with timestamp."""
        episodic = state["episodic_memory"].copy()
        timestamp = datetime.now().strftime("%H:%M:%S")
        episodic.append(f"[{timestamp}] {observation[:100]}")
        
        if len(episodic) > MemoryCascade.EPISODIC_CAPACITY:
            episodic = episodic[-MemoryCascade.EPISODIC_CAPACITY:]
        
        return {"episodic_memory": episodic, "observation": observation}


# ======================================================================
# STATE DEFINITION
# ======================================================================

class OrchestratorState(TypedDict):
    """State for the orchestrator subagent."""
    messages: Annotated[List[BaseMessage], operator.add]
    working_memory: str
    episodic_memory: List[str]
    semantic_memory: str
    next_action: str
    observation: str
    loop_count: int
    is_background: bool


# ======================================================================
# NODES
# ======================================================================

def reasoning_node(state: OrchestratorState, llm, tools_list: List[BaseTool], base_prompt: str) -> dict:
    """
    LLM-driven reasoning: decides tool use, kairos, or final answer.
    """
    system_prompt = MemoryCascade.build_system_prompt(state, base_prompt)
    llm_with_tools = llm.bind_tools(tools_list)
    
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )
    
    next_action = "final"
    if isinstance(response, AIMessage) and response.tool_calls:
        next_action = "tool"
    
    working_mem = f"LLM response: {response.content if response.content else 'tool call'}"
    
    return {
        "messages": [response],
        "next_action": next_action,
        "working_memory": working_mem,
        "loop_count": state.get("loop_count", 0) + 1,
    }


def security_gate_node(state: OrchestratorState) -> dict:
    """
    Validate tool calls before execution.
    Rejects dangerous operations.
    """
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    tool_calls = last_message.tool_calls
    rejected_calls = []
    
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        
        if tool_name == "bash" and any(
            dangerous in str(args).lower()
            for dangerous in ["rm -rf /", "sudo", "dd if=/dev/"]
        ):
            rejected_calls.append({
                "call": call,
                "reason": f"Dangerous bash command rejected: {tool_name}"
            })
    
    if rejected_calls:
        rejection_messages = []
        for rejected in rejected_calls:
            rejection_messages.append(
                ToolMessage(
                    content=rejected["reason"],
                    tool_call_id=rejected["call"]["id"],
                    name=rejected["call"]["name"],
                )
            )
        return {"messages": rejection_messages}
    
    return {}


def tool_execution_node(state: OrchestratorState, tools_list: List[BaseTool]) -> dict:
    """
    Execute tool calls using LangChain's ToolNode.
    """
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    tool_node = ToolNode(tools=tools_list)
    
    try:
        tool_results = tool_node.invoke({"messages": state["messages"]})
        tool_messages = tool_results.get("messages", [])
        
    except Exception as e:
        last_call = last_message.tool_calls[0]
        tool_messages = [
            ToolMessage(
                content=f"Tool execution error: {str(e)[:200]}",
                tool_call_id=last_call["id"],
                name=last_call["name"],
            )
        ]
    
    # Integrate into episodic memory
    updated_episodic = state["episodic_memory"].copy()
    for msg in tool_messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        updated_episodic.append(f"[{timestamp}] {msg.name}: {msg.content}")
    
    if len(updated_episodic) > MemoryCascade.EPISODIC_CAPACITY:
        updated_episodic = updated_episodic[-MemoryCascade.EPISODIC_CAPACITY:]
    
    return {
        "messages": tool_messages,
        "episodic_memory": updated_episodic,
        "observation": tool_messages[0].content if tool_messages else "",
    }


def context_compaction_node(state: OrchestratorState, llm) -> dict:
    """
    Periodically distill episodic memory into semantic memory.
    """
    loop_count = state.get("loop_count", 0)
    should_compact = (
        len(state["episodic_memory"]) >= MemoryCascade.EPISODIC_CAPACITY
        or loop_count % 20 == 0
    )
    
    if not should_compact:
        return {}
    
    episodic = state["episodic_memory"]
    if not episodic:
        return {}
    
    episodic_str = "\n".join(episodic[-10:])
    
    try:
        summary_prompt = f"""Summarize these observations into 2-3 key facts:

{episodic_str}

Summary:"""
        summary_response = llm.invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
    except Exception as e:
        summary = "; ".join(episodic[-10:])
    
    new_semantic = state["semantic_memory"] + f"\n[Loop {loop_count}] {summary}"
    
    if len(new_semantic) > MemoryCascade.SEMANTIC_MAX_TOKENS:
        new_semantic = new_semantic[-MemoryCascade.SEMANTIC_MAX_TOKENS:]
    
    return {
        "semantic_memory": new_semantic,
        "episodic_memory": episodic[-10:],
    }


# ======================================================================
# ROUTING
# ======================================================================

def route_after_reasoning(state: OrchestratorState) -> str:
    """Route based on next_action."""
    next_action = state.get("next_action", "final")
    
    if next_action == "tool":
        return "security_gate"
    else:
        return END


def route_after_tool(state: OrchestratorState) -> str:
    """After tool execution, compact and return to reasoning."""
    return "context_compaction"


# ======================================================================
# GRAPH BUILD
# ======================================================================

def build_orchestrator_graph(llm, tools_list: List[BaseTool], base_prompt: str):
    """Assemble the orchestrator graph."""
    workflow = StateGraph(OrchestratorState)
    
    from functools import partial
    reasoning_with_context = partial(
        reasoning_node, llm=llm, tools_list=tools_list, base_prompt=base_prompt
    )
    tool_execution_with_tools = partial(tool_execution_node, tools_list=tools_list)
    context_compaction_with_llm = partial(context_compaction_node, llm=llm)
    
    workflow.add_node("reasoning", reasoning_with_context)
    workflow.add_node("security_gate", security_gate_node)
    workflow.add_node("tool_execution", tool_execution_with_tools)
    workflow.add_node("context_compaction", context_compaction_with_llm)
    
    workflow.set_entry_point("reasoning")
    
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {"security_gate": "security_gate", END: END},
    )
    
    workflow.add_edge("security_gate", "tool_execution")
    workflow.add_edge("tool_execution", "context_compaction")
    workflow.add_edge("context_compaction", "reasoning")
    
    return workflow.compile()


# ======================================================================
# MAIN EXECUTION
# ======================================================================

def run_orchestrator(task: str, instructions: str = "Return a concise summary with findings.") -> str:
    """
    Run the orchestrator as an isolated subagent.
    
    Args:
        task: The main task/query to execute
        instructions: Output format instructions for the final response
    
    Returns:
        Clean summary of the orchestrator's work
    """
    try:
        from setup import system_prompt
    except ImportError:
        system_prompt = "You are a helpful AI assistant with access to tools."
    
    # Assemble tools
    orch_tools = task_state_tools + base_tools + [ask_clarifying_questions_tool, parse_pdf,str_replace_tool] 
    
    # Initialize LLM
    orch_llm = ChatOllama(
        model=ORCHESTRATOR_MODEL,
        base_url="http://localhost:11434",
        num_ctx=ORCHESTRATOR_CTX_WINDOW,
        stream=False
    )
    
    # Build graph
    app = build_orchestrator_graph(orch_llm, orch_tools, system_prompt)
    
    # Initialize state
    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=task)],
        "working_memory": "",
        "episodic_memory": [],
        "semantic_memory": "",
        "next_action": "",
        "observation": "",
        "loop_count": 0,
        "is_background": False,
    }
    
    # Run the graph with iteration limit
    final_state = None
    iteration = 0
    max_iterations = 15
    
    try:
        for state_update in app.stream(initial_state, config={"recursion_limit": 100}):
            for node_name, node_output in state_update.items():
                # Merge updates
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if key == "messages" and isinstance(initial_state.get("messages"), list):
                            if isinstance(value, list):
                                initial_state["messages"].extend(value)
                            else:
                                initial_state["messages"].append(value)
                        else:
                            initial_state[key] = value
            
            iteration += 1
            if iteration >= max_iterations or initial_state.get("next_action") == "final":
                break
        
        final_state = initial_state
        
    except Exception as e:
        console.print(f"[red]Orchestrator error: {str(e)[:200]}[/red]")
        return f"Orchestrator failed: {str(e)[:200]}"
    
    # Extract final response from last AI message
    last_message = final_state["messages"][-1]
    result = ""
    
    if isinstance(last_message, AIMessage) and last_message.content:
        result = str(last_message.content)
    else:
        # Fallback: summarize observations
        if final_state["episodic_memory"]:
            result = "\n".join(final_state["episodic_memory"][-10:])
        else:
            result = "Orchestrator completed but no output generated."
    
    # Log token usage if available
    if hasattr(last_message, 'response_metadata') and last_message.response_metadata:
        meta = last_message.response_metadata
        turn_tokens = (meta.get("prompt_eval_count") or 0) + (meta.get("eval_count") or 0)
        # console.print(f"[dim]Tokens: {turn_tokens}[/dim]")
    
    console.print("[bold green]✓ Subagent Done[/bold green]")
    return result


# ======================================================================
# WRAP AS LANGCHAIN TOOL
# ======================================================================

def _run_orchestrator_wrapper(task: str) -> str:
    """Wrapper for LangChain Tool."""
    console.print(_BULLET," [bold cyan]Subagent running...[/bold cyan]")
    console.print(_NEST, f"[dim]Prompt:{task:100}[/dim]")
    return run_orchestrator(task)


subagent_tool = Tool(
    name="run_orchestrator",
    func=_run_orchestrator_wrapper,
    description=(
        "Delegate complex, multi-step tasks to the main orchestrator agent. "
        "Use this for tasks requiring reasoning, tool chaining, file analysis, or context management. "
        "The orchestrator manages its own memory cascade (working, episodic, semantic) "
        "and can execute multiple tool calls in sequence. "
        "Returns a clean summary of the results. "
        "Input: a clear description of the task to complete."
    )
)


# ======================================================================
# OPTIONAL: INTEGRATE INTO YOUR AGENT
# ======================================================================

if __name__ == "__main__":
    # Test the orchestrator as a tool
    test_task = "List all files in the current directory and summarize what you find."
    print("Testing orchestrator tool...")
    result = run_orchestrator(test_task)
    print("\n=== RESULT ===")
    print(result)