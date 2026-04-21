# ======================================================================||
# ORCHESTRATOR AS SUBAGENT TOOL - Enhanced Memory Architecture
# ======================================================================||
# Full three-tier memory cascade system matching main agent.
# Intelligent distillation, caching, and compaction triggers.
# ======================================================================||

import operator
import sys
import time
from dotenv import load_dotenv
import os
from typing import Annotated, TypedDict, Union, List, Optional, Sequence
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import asyncio

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool, Tool, tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from rich.console import Console

from setup import workspace
from basic_tools import base_tools
from clarification_tool import ask_clarifying_questions_tool
from task_state_tool import task_state_tools

console = Console()

# ======================================================================
# CONFIGURATION
# ======================================================================
_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"

ORCHESTRATOR_MODEL = "gemma4:31b-cloud"
ORCHESTRATOR_CTX_WINDOW = 256000
ORCHESTRATOR_CHAR_DELAY = 0.01

load_dotenv()
API = os.getenv("Parser_API")

# ======================================================================
# ASYNC PDF PARSING TOOL
# ======================================================================

async def _parse_pdf_async(file_path: str, tier: str = "agentic") -> str:
    """Async PDF parsing using Llama Cloud API."""
    try:
        from llama_cloud import AsyncLlamaCloud
        _llama_client = AsyncLlamaCloud(api_key=API)
        file_obj = await _llama_client.files.create(file=file_path, purpose="parse")
        result = await _llama_client.parsing.parse(
            file_id=file_obj.id,
            tier=tier,
            version="latest",
            expand=["markdown_full", "text_full"],
        )
        return result.markdown_full or result.text_full or "No content extracted"
    except Exception as e:
        return f"PDF parsing failed: {str(e)}"


@tool
def parse_pdf(file_path: str, tier: str = "agentic") -> str:
    """Parse a PDF and return its content as markdown.
    file_path is relative to workspace (e.g. 'report.pdf' or 'papers/doc.pdf')
    tier options: fast, cost_effective, agentic, agentic_plus
    """
    full_path = Path(workspace) / file_path
    if not full_path.exists():
        full_path = Path(file_path)
    if not full_path.exists():
        return f"❌ File not found: {file_path}"
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_parse_pdf_async(str(full_path), tier))
        finally:
            loop.close()
    except Exception as e:
        return f"❌ PDF parsing failed: {str(e)}"


# ======================================================================
# ENHANCED MEMORY CASCADE - Full Architecture
# ======================================================================

class CompactionStrategy(Enum):
    """Strategies for semantic memory distillation."""
    FIFO = "fifo"
    SEMANTIC_DISTILL = "semantic_distill"
    TOPIC_CLUSTERING = "topic_clustering"


class SubagentMemoryCascade:
    """
    Full three-tier memory system for subagent orchestrator.
    Mirrors the main agent's architecture with intelligent distillation.
    
    - Episodic Memory: Recent raw observations (FIFO eviction when full)
    - Semantic Memory: Distilled long-term knowledge (LLM-summarized, persistent)
    - Working Memory: Current turn state (ephemeral)
    
    Compaction Strategy:
    - When episodic exceeds capacity, distill oldest N observations → semantic
    - Semantic memory preserves context while reducing token overhead
    - Caching prevents redundant LLM calls for same observations
    """
    
    # Capacity tuning
    EPISODIC_CAPACITY = 10
    EPISODIC_BATCH_SIZE = 3
    SEMANTIC_MAX_CHARS = 2000
    COMPACTION_INTERVAL = 20
    
    # Strategy selection
    DEFAULT_STRATEGY = CompactionStrategy.SEMANTIC_DISTILL
    
    # Base system prompt
    BASE_SYSTEM_PROMPT = None
    
    # Distillation cache
    _distillation_cache: dict = {}
    
    @staticmethod
    def set_base_prompt(prompt: str):
        """Set the base system prompt."""
        SubagentMemoryCascade.BASE_SYSTEM_PROMPT = prompt
    
    @staticmethod
    def _hash_observations(obs_list: List[str]) -> str:
        """Create a hash of observations for caching."""
        content = "|".join(obs_list)
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _distill_observations_with_llm(observations: List[str], llm) -> str:
        """
        Use LLM to intelligently summarize observations.
        Caches results to avoid redundant API calls.
        """
        obs_hash = SubagentMemoryCascade._hash_observations(observations)
        
        # Check cache
        if obs_hash in SubagentMemoryCascade._distillation_cache:
            return SubagentMemoryCascade._distillation_cache[obs_hash]
        
        episodic_str = "\n".join(observations)
        
        try:
            summary_prompt = f"""You are extracting key facts from a conversation.
Distill the following observations into 2-3 concise bullet points for long-term memory.
Focus on: what was done, what was learned, and current state.

Observations:
{episodic_str}

Provide ONLY the 2-3 bullet points, no preamble:"""
            
            response = llm.invoke(summary_prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            
            # Cache the result
            SubagentMemoryCascade._distillation_cache[obs_hash] = summary
            return summary
            
        except Exception as e:
            # Fallback: simple concatenation
            fallback = "; ".join([f"{obs[:50]}..." for obs in observations])
            SubagentMemoryCascade._distillation_cache[obs_hash] = fallback
            return fallback
    
    @staticmethod
    def should_trigger_compaction(state: 'OrchestratorState') -> bool:
        """
        Determine if episodic→semantic compaction should occur.
        
        Triggers:
        1. Episodic memory exceeds capacity
        2. Loop count hits compaction interval
        """
        loop_count = state.get("loop_count", 0)
        episodic_len = len(state.get("episodic_memory", []))
        
        return (
            episodic_len >= SubagentMemoryCascade.EPISODIC_CAPACITY
            or (loop_count > 0 and loop_count % SubagentMemoryCascade.COMPACTION_INTERVAL == 0)
        )
    
    @staticmethod
    def compact_memory(state: 'OrchestratorState', llm, strategy: CompactionStrategy = None) -> dict:
        """
        Distill episodic memory into semantic memory.
        
        Process:
        1. Select oldest batch from episodic memory
        2. Summarize using LLM (if strategy permits)
        3. Append to semantic memory with metadata
        4. Evict from episodic memory
        5. Trim semantic if exceeds limit
        """
        if strategy is None:
            strategy = SubagentMemoryCascade.DEFAULT_STRATEGY
        
        episodic = state.get("episodic_memory", []).copy()
        semantic = state.get("semantic_memory", "")
        
        if not episodic:
            return {}
        
        # Select batch to distill (oldest BATCH_SIZE items)
        batch_to_distill = episodic[:SubagentMemoryCascade.EPISODIC_BATCH_SIZE]
        remaining_episodic = episodic[SubagentMemoryCascade.EPISODIC_BATCH_SIZE:]
        
        # Distill based on strategy
        if strategy == CompactionStrategy.SEMANTIC_DISTILL:
            distilled = SubagentMemoryCascade._distill_observations_with_llm(batch_to_distill, llm)
        elif strategy == CompactionStrategy.FIFO:
            distilled = "; ".join(batch_to_distill)
        else:
            distilled = "; ".join(batch_to_distill)
        
        # Append to semantic memory with metadata
        loop_count = state.get("loop_count", 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp} | Loop {loop_count}]\n{distilled}\n"
        
        new_semantic = semantic + "\n" + new_entry
        
        # Trim semantic memory if exceeds limit
        if len(new_semantic) > SubagentMemoryCascade.SEMANTIC_MAX_CHARS:
            new_semantic = new_semantic[-SubagentMemoryCascade.SEMANTIC_MAX_CHARS:]
        
        return {
            "episodic_memory": remaining_episodic,
            "semantic_memory": new_semantic,
        }
    
    @staticmethod
    def build_system_prompt(state: 'OrchestratorState', base_prompt: str) -> str:
        """
        Construct the system prompt that injects all three memory tiers.
        """
        base = base_prompt
        memory_sections = []
        
        # Add semantic memory (long-term context) first
        if state.get("semantic_memory"):
            semantic_section = f"""=== LONG-TERM PROJECT CONTEXT ===
{state['semantic_memory']}

"""
            memory_sections.append(semantic_section)
        
        # Add episodic memory (recent observations)
        if state.get("episodic_memory"):
            recent_obs = state["episodic_memory"][-5:]
            episodic_section = """=== RECENT OBSERVATIONS ===
"""
            for obs in recent_obs:
                episodic_section += f"• {obs}\n"
            episodic_section += "\n"
            memory_sections.append(episodic_section)
        
        # Combine all sections
        memory_context = "".join(memory_sections)
        system = f"""{base}

{memory_context}"""
        
        return system
    
    @staticmethod
    def add_observation(state: 'OrchestratorState', observation: str) -> dict:
        """
        After tool execution, integrate observation into episodic memory.
        Enforces EPISODIC_CAPACITY limit via FIFO eviction.
        """
        episodic = state.get("episodic_memory", []).copy()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Truncate very long observations
        if len(observation) > 300:
            observation = observation[:300] + "..."
        
        episodic.append(f"[{timestamp}] {observation}")
        
        # Enforce capacity: keep only most recent N observations
        if len(episodic) > SubagentMemoryCascade.EPISODIC_CAPACITY:
            episodic = episodic[-SubagentMemoryCascade.EPISODIC_CAPACITY:]
        
        return {
            "episodic_memory": episodic,
            "observation": observation,
        }


# ======================================================================
# STATE DEFINITION
# ======================================================================

class OrchestratorState(TypedDict):
    """State for the orchestrator subagent with full memory cascade."""
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
    LLM-driven reasoning with memory injection.
    Decides: tool use, kairos, or final answer.
    """
    system_prompt = SubagentMemoryCascade.build_system_prompt(state, base_prompt)
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
    Integrates observations into episodic memory.
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
    updated_episodic = state.get("episodic_memory", []).copy()
    for msg in tool_messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        content = msg.content if len(msg.content) <= 150 else msg.content[:150] + "..."
        updated_episodic.append(f"[{timestamp}] {msg.name}: {content}")
    
    # Enforce episodic capacity
    if len(updated_episodic) > SubagentMemoryCascade.EPISODIC_CAPACITY:
        updated_episodic = updated_episodic[-SubagentMemoryCascade.EPISODIC_CAPACITY:]
    
    return {
        "messages": tool_messages,
        "episodic_memory": updated_episodic,
        "observation": tool_messages[0].content if tool_messages else "",
    }


def context_compaction_node(state: OrchestratorState, llm) -> dict:
    """
    Periodically distill episodic memory into semantic memory.
    Intelligent triggers: capacity overflow or periodic interval.
    """
    if not SubagentMemoryCascade.should_trigger_compaction(state):
        return {}
    
    episodic = state.get("episodic_memory", [])
    if not episodic:
        return {}
    
    # Perform compaction with intelligent distillation
    compaction_result = SubagentMemoryCascade.compact_memory(
        state,
        llm,
        strategy=CompactionStrategy.SEMANTIC_DISTILL
    )
    
    return compaction_result


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


# ======================================================================
# GRAPH BUILD
# ======================================================================

def build_orchestrator_graph(llm, tools_list: List[BaseTool], base_prompt: str):
    """Assemble the orchestrator graph with full memory architecture."""
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
    orch_tools = task_state_tools + base_tools + [ask_clarifying_questions_tool, parse_pdf]
    
    # Initialize LLM
    orch_llm = ChatOllama(
        model=ORCHESTRATOR_MODEL,
        base_url="http://localhost:11434",
        num_ctx=ORCHESTRATOR_CTX_WINDOW,
        stream=False
    )
    
    # Set base prompt for memory system
    SubagentMemoryCascade.set_base_prompt(system_prompt)
    
    # Build graph
    app = build_orchestrator_graph(orch_llm, orch_tools, system_prompt)
    
    # Initialize state with full memory cascade
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
        if final_state["episodic_memory"]:
            result = "\n".join(final_state["episodic_memory"][-10:])
        else:
            result = "Orchestrator completed but no output generated."
    
    console.print("[bold green]✓ Subagent Done[/bold green]")
    return result


# ======================================================================
# WRAP AS LANGCHAIN TOOL
# ======================================================================

def _run_orchestrator_wrapper(task: str) -> str:
    """Wrapper for LangChain Tool."""
    console.print(_BULLET, "[bold cyan]Subagent running...[/bold cyan]")
    console.print(_NEST, f"[dim]Prompt: {task[:100]}...[/dim]")
    return run_orchestrator(task)


subagent_tool = Tool(
    name="run_orchestrator",
    func=_run_orchestrator_wrapper,
    description=(
        "Delegate complex, multi-step tasks to the orchestrator subagent. "
        "Use this for tasks requiring reasoning, tool chaining, file analysis, or context management. "
        "The orchestrator manages a full three-tier memory cascade: working memory, episodic memory with FIFO eviction, "
        "and semantic memory with intelligent LLM-based distillation. "
        "This allows it to handle long-running tasks while maintaining context efficiency. "
        "Returns a clean summary of the results. "
        "Input: a clear description of the task to complete."
    )
)


# ======================================================================
# OPTIONAL: TESTING
# ======================================================================

if __name__ == "__main__":
    test_task = "List all files in the current directory and summarize what you find."
    print("Testing enhanced orchestrator subagent...")
    result = run_orchestrator(test_task)
    print("\n=== RESULT ===")
    print(result)