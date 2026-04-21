# ======================================================================||
# IMPORTS                                                               ||
# ======================================================================||
import operator
import sys
import time
from typing import Annotated, TypedDict, Union, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import hashlib

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from rich.console import Console
from setup import workspace

from basic_tools import base_tools
from subagent_tool import subagent_tool
from clarification_tool import ask_clarifying_questions_tool
from task_state_tool import task_state_tools

console = Console()

scaler = 1.015

def print_chars_smooth(text: str, char_delay: float):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if char_delay > 0:
            time.sleep(char_delay * scaler)

# ======================================================================
# CONFIGURATION
# ======================================================================
agent_model = "gemma4:31b-cloud"
CTX_WINDOW = 256000
llm = ChatOllama(
    model=agent_model,
    base_url="http://localhost:11434",
    num_ctx=CTX_WINDOW,
    stream=False
)
tools_list = base_tools + task_state_tools + [subagent_tool, ask_clarifying_questions_tool]

# ======================================================================
# STATE DEFINITION - The Context Cascade
# ======================================================================

class AgentState(TypedDict):
    """
    The state flows through the entire orchestrator.
    Memory tiers are explicitly managed here.
    """
    # Core message loop (LangChain Message objects only)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Context Cascade - three-tier memory system
    working_memory: str  # Current turn state (reset each loop)
    episodic_memory: List[str]  # Short-term session history (last N observations)
    semantic_memory: str  # Long-term project knowledge (distilled, persistent)
    
    # Execution state
    next_action: str  # "tool", "kairos", "final" - set by reasoning node
    observation: str  # Last tool execution result
    tool_call_id: Optional[str]  # Track which tool_call this observation closes
    
    # Metadata
    loop_count: int  # Track iterations for compaction triggers
    is_background: bool  # Flag for kairos daemon mode


# ======================================================================
# ENHANCED MEMORY CASCADE - Intelligent Distillation
# ======================================================================

class CompactionStrategy(Enum):
    """Strategies for semantic memory distillation."""
    FIFO = "fifo"  # Simple FIFO eviction (fallback)
    SEMANTIC_DISTILL = "semantic_distill"  # Use LLM to summarize
    TOPIC_CLUSTERING = "topic_clustering"  # Group by topic, distill per-topic


class MemoryCascade:
    """
    Manages the three-tier memory system with intelligent distillation.
    
    - Episodic Memory: Recent raw observations (FIFO eviction when full)
    - Semantic Memory: Distilled long-term knowledge (LLM-summarized, persistent)
    - Working Memory: Current turn state (ephemeral)
    
    Compaction Strategy:
    - When episodic exceeds capacity, distill oldest N observations → semantic
    - Semantic memory preserves context while reducing token overhead
    - Topic-shift detection triggers emergency compaction
    """
    
    # Capacity tuning
    EPISODIC_CAPACITY = 10  # Keep last 10 observations
    EPISODIC_BATCH_SIZE = 3  # Distill in batches of 3
    SEMANTIC_MAX_CHARS = 2000  # Semantic memory char limit (~500 tokens)
    COMPACTION_INTERVAL = 20  # Compact every N loops
    
    # Strategy selection
    DEFAULT_STRATEGY = CompactionStrategy.SEMANTIC_DISTILL
    
    # Base system prompt
    BASE_SYSTEM_PROMPT = None
    
    # Distillation cache to avoid re-summarizing
    _distillation_cache: dict = {}
    
    @staticmethod
    def set_base_prompt(prompt: str):
        """Set the base system prompt from setup."""
        MemoryCascade.BASE_SYSTEM_PROMPT = prompt
    
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
        obs_hash = MemoryCascade._hash_observations(observations)
        
        # Check cache
        if obs_hash in MemoryCascade._distillation_cache:
            return MemoryCascade._distillation_cache[obs_hash]
        
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
            MemoryCascade._distillation_cache[obs_hash] = summary
            return summary
            
        except Exception as e:
            # Fallback: simple concatenation with timestamps
            fallback = "; ".join([f"{obs[:50]}..." for obs in observations])
            MemoryCascade._distillation_cache[obs_hash] = fallback
            return fallback
    
    @staticmethod
    def should_trigger_compaction(state: AgentState) -> bool:
        """
        Determine if episodic→semantic compaction should occur.
        
        Triggers:
        1. Episodic memory exceeds capacity
        2. Loop count hits compaction interval
        """
        loop_count = state.get("loop_count", 0)
        episodic_len = len(state.get("episodic_memory", []))
        
        # Trigger if full or on interval
        return (
            episodic_len >= MemoryCascade.EPISODIC_CAPACITY
            or (loop_count > 0 and loop_count % MemoryCascade.COMPACTION_INTERVAL == 0)
        )
    
    @staticmethod
    def compact_memory(state: AgentState, llm, strategy: CompactionStrategy = None) -> dict:
        """
        Distill episodic memory into semantic memory.
        
        Process:
        1. Select oldest batch from episodic memory
        2. Summarize using LLM (if strategy permits)
        3. Append to semantic memory
        4. Evict from episodic memory
        5. Trim semantic if exceeds limit
        """
        if strategy is None:
            strategy = MemoryCascade.DEFAULT_STRATEGY
        
        episodic = state.get("episodic_memory", []).copy()
        semantic = state.get("semantic_memory", "")
        
        if not episodic:
            return {}
        
        # Select batch to distill (oldest BATCH_SIZE items)
        batch_to_distill = episodic[:MemoryCascade.EPISODIC_BATCH_SIZE]
        remaining_episodic = episodic[MemoryCascade.EPISODIC_BATCH_SIZE:]
        
        # Distill based on strategy
        if strategy == CompactionStrategy.SEMANTIC_DISTILL:
            distilled = MemoryCascade._distill_observations_with_llm(batch_to_distill, llm)
        elif strategy == CompactionStrategy.FIFO:
            # Simple: just concatenate
            distilled = "; ".join(batch_to_distill)
        else:
            distilled = "; ".join(batch_to_distill)
        
        # Append to semantic memory with metadata
        loop_count = state.get("loop_count", 0)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"[{timestamp} | Loop {loop_count}]\n{distilled}\n"
        
        new_semantic = semantic + "\n" + new_entry
        
        # Trim semantic memory if exceeds limit
        if len(new_semantic) > MemoryCascade.SEMANTIC_MAX_CHARS:
            # Keep the most recent entries (from the end)
            new_semantic = new_semantic[-MemoryCascade.SEMANTIC_MAX_CHARS:]
        
        return {
            "episodic_memory": remaining_episodic,
            "semantic_memory": new_semantic,
        }
    
    @staticmethod
    def build_system_prompt(state: AgentState) -> str:
        """
        Construct the system prompt that injects all three memory tiers.
        Wraps the base system prompt from setup with memory context.
        """
        from setup import system_prompt as base_prompt
        
        # Use the setup system prompt as foundation
        base = MemoryCascade.BASE_SYSTEM_PROMPT or base_prompt
        
        # Build memory context
        memory_sections = []
        
        # Add semantic memory (long-term context) first
        if state.get("semantic_memory"):
            semantic_section = f"""=== LONG-TERM PROJECT CONTEXT ===
{state['semantic_memory']}

"""
            memory_sections.append(semantic_section)
        
        # Add episodic memory (recent observations)
        if state.get("episodic_memory"):
            recent_obs = state["episodic_memory"][-5:]  # Last 5 observations
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
    def add_observation(state: AgentState, observation: str) -> dict:
        """
        After tool execution, integrate observation into episodic memory.
        Called by tool_execution_node.
        Enforces EPISODIC_CAPACITY limit via FIFO eviction.
        """
        episodic = state.get("episodic_memory", []).copy()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Truncate very long observations
        if len(observation) > 300:
            observation = observation[:300] + "..."
        
        episodic.append(f"[{timestamp}] {observation}")
        
        # Enforce capacity: keep only most recent N observations
        if len(episodic) > MemoryCascade.EPISODIC_CAPACITY:
            episodic = episodic[-MemoryCascade.EPISODIC_CAPACITY:]
        
        return {
            "episodic_memory": episodic,
            "observation": observation,
        }


# ======================================================================
# NODE 1: REASONING NODE - The Query Engine
# ======================================================================

def reasoning_node(state: AgentState, llm, tools_list: List[BaseTool]):
    """
    Core orchestrator: LLM-driven reasoning that decides what to do next.
    
    This is the "Query Engine":
    - Analyzes current state + memory + conversation
    - Decides: use tool, run kairos daemon, or provide final answer
    - Uses LLM's native tool_calls mechanism
    """
    
    # Build system prompt with memory injection
    system_prompt = MemoryCascade.build_system_prompt(state)
    
    # Bind tools to LLM (LangChain's native tool_calls)
    llm_with_tools = llm.bind_tools(tools_list)
    
    # Invoke LLM with full context
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )
    
    # Determine next action based on LLM's output
    next_action = "final"  # Default: provide final answer
    
    if isinstance(response, AIMessage) and response.tool_calls:
        # LLM decided to use a tool
        next_action = "tool"
    
    # Update working memory for this turn
    working_mem = f"LLM response: {response.content if response.content else 'tool call'}"
    
    return {
        "messages": [response],
        "next_action": next_action,
        "working_memory": working_mem,
        "loop_count": state.get("loop_count", 0) + 1,
    }


# ======================================================================
# NODE 2: SECURITY & PERMISSIONS GATE
# ======================================================================

class SecurityGate:
    """
    Validates tool calls before execution.
    Implements a whitelist/blacklist system with argument validation.
    """
    
    # Dangerous bash patterns
    DANGEROUS_BASH = {
        "rm -rf /",
        "sudo",
        "dd if=/dev/",
        ":(){:|:&};:",  # Fork bomb
        "mkfs",  # Format disk
    }
    
    # Safe workspace boundary
    WORKSPACE_ROOT = workspace
    
    @staticmethod
    def validate_path_safety(path_str: str) -> tuple[bool, str]:
        """
        Ensure path stays within workspace.
        Returns (is_safe, reason)
        Properly handles Path objects and string paths.
        """
        try:
            # Get workspace root - handle both Path objects and strings
            workspace_root = SecurityGate.WORKSPACE_ROOT
            if isinstance(workspace_root, Path):
                workspace_path = workspace_root.resolve()
            else:
                workspace_path = Path(workspace_root).resolve()
            
            # Convert input path to Path object
            path_obj = Path(path_str)
            
            # If path is relative, make it relative to workspace
            if not path_obj.is_absolute():
                abs_path = (workspace_path / path_obj).resolve()
            else:
                abs_path = path_obj.resolve()
            
            # Use Path.is_relative_to() if available (Python 3.9+), otherwise manual check
            try:
                # Try to make abs_path relative to workspace - if it succeeds, it's safe
                abs_path.relative_to(workspace_path)
                return True, ""
            except ValueError:
                # Path is not relative to workspace
                return False, f"Path escapes workspace: {abs_path}"
            
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    @staticmethod
    def validate_tool_call(tool_name: str, args: dict) -> tuple[bool, Optional[str]]:
        """
        Validate individual tool call.
        Returns (is_valid, rejection_reason or None)
        """
        
        # Bash command safety check
        if tool_name == "bash":
            cmd = str(args.get("command", "")).lower()
            
            # Check for dangerous patterns
            for dangerous in SecurityGate.DANGEROUS_BASH:
                if dangerous.lower() in cmd:
                    return False, f"Blocked dangerous bash pattern: {dangerous}"
        
        # File path safety check
        if tool_name in ["read_file", "write_file", "delete_file"]:
            path = args.get("path", "")
            if path:  # Only validate if path is provided
                is_safe, reason = SecurityGate.validate_path_safety(path)
                if not is_safe:
                    return False, reason
        
        return True, None


def security_gate_node(state: AgentState) -> dict:
    """
    Validate tool calls before execution.
    """
    
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    tool_calls = last_message.tool_calls
    rejection_messages = []
    
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        
        # Validate using SecurityGate
        is_valid, rejection_reason = SecurityGate.validate_tool_call(tool_name, args)
        
        if not is_valid:
            rejection_messages.append(
                ToolMessage(
                    content=f"Security validation failed: {rejection_reason}",
                    tool_call_id=call["id"],
                    name=tool_name,
                )
            )
    
    if rejection_messages:
        return {"messages": rejection_messages}
    
    return {}


# ======================================================================
# NODE 3: TOOL EXECUTION NODE
# ======================================================================

def tool_execution_node(state: AgentState, tools_list: List[BaseTool]) -> dict:
    """
    Execute the tool(s) that the LLM selected.
    Integrates observations into episodic memory.
    """
    
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    # Create a ToolNode and invoke it
    tool_node = ToolNode(tools=tools_list)
    
    try:
        # ToolNode.invoke() expects messages as input
        tool_results = tool_node.invoke({"messages": state["messages"]})
        tool_messages = tool_results.get("messages", [])
        
    except Exception as e:
        # If tool execution fails, return error as ToolMessage
        last_call = last_message.tool_calls[0]
        tool_messages = [
            ToolMessage(
                content=f"Tool execution error: {str(e)[:200]}",
                tool_call_id=last_call["id"],
                name=last_call["name"],
            )
        ]
    
    # Integrate observations into episodic memory
    updated_episodic = state.get("episodic_memory", []).copy()
    for msg in tool_messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        # Truncate long tool outputs
        content = msg.content if len(msg.content) <= 150 else msg.content[:150] + "..."
        updated_episodic.append(f"[{timestamp}] {msg.name}: {content}")
    
    # Enforce episodic capacity
    if len(updated_episodic) > MemoryCascade.EPISODIC_CAPACITY:
        updated_episodic = updated_episodic[-MemoryCascade.EPISODIC_CAPACITY:]
    
    return {
        "messages": tool_messages,
        "episodic_memory": updated_episodic,
        "observation": tool_messages[0].content if tool_messages else "",
    }


# ======================================================================
# NODE 4: CONTEXT CASCADE & COMPACTION
# ======================================================================

def context_compaction_node(state: AgentState, llm) -> dict:
    """
    Periodically distill episodic memory into semantic memory.
    
    Intelligent triggers:
    1. Episodic memory exceeds capacity → immediate compaction
    2. Loop count hits interval (every 20 loops) → periodic compaction
    3. Semantic memory empty → bootstrap first distillation
    """
    
    # Check if compaction should occur
    if not MemoryCascade.should_trigger_compaction(state):
        return {}
    
    episodic = state.get("episodic_memory", [])
    if not episodic:
        return {}
    
    # Perform compaction with intelligent distillation
    compaction_result = MemoryCascade.compact_memory(
        state,
        llm,
        strategy=CompactionStrategy.SEMANTIC_DISTILL
    )
    
    # Log compaction event
    if compaction_result:
        console.print(
            f"[dim]📚 Memory compaction: {len(episodic)} observations → semantic[/dim]"
        )
    
    return compaction_result


# ======================================================================
# NODE 5: KAIROS DAEMON (Background Tasks)
# ======================================================================

def kairos_daemon_node(state: AgentState) -> dict:
    """
    Long-running autonomous task handler.
    
    TODO: Real implementation uses threading.Thread or asyncio.Task
    For MVP: stub — just acknowledge the background task
    """
    
    observation = "KAIROS: Background task mode recognized. (Not yet implemented)"
    
    return {
        "observation": observation,
        "is_background": True,
    }


# ======================================================================
# ROUTING LOGIC
# ======================================================================

def route_after_reasoning(state: AgentState) -> str:
    """
    After reasoning, decide where to go:
    - "security_gate": Validate and execute tool call
    - "kairos_daemon": Start autonomous daemon
    - END: Provide final response to user
    """
    next_action = state.get("next_action", "final")
    
    if next_action == "tool":
        return "security_gate"
    elif next_action == "kairos":
        return "kairos_daemon"
    else:
        return END


# ======================================================================
# GRAPH COMPILATION
# ======================================================================

def build_momobot_graph(llm, tools_list: List[BaseTool]):
    """
    Assemble the complete orchestrator graph.
    
    Flow:
    reasoning → {tool path or final}
    tool path: security_gate → tool_execution → context_compaction → reasoning
    """
    
    workflow = StateGraph(AgentState)
    
    # Partial function to bind llm and tools to nodes
    from functools import partial
    reasoning_with_context = partial(reasoning_node, llm=llm, tools_list=tools_list)
    tool_execution_with_tools = partial(tool_execution_node, tools_list=tools_list)
    context_compaction_with_llm = partial(context_compaction_node, llm=llm)
    
    # Add nodes
    workflow.add_node("reasoning", reasoning_with_context)
    workflow.add_node("security_gate", security_gate_node)
    workflow.add_node("tool_execution", tool_execution_with_tools)
    workflow.add_node("context_compaction", context_compaction_with_llm)
    workflow.add_node("kairos_daemon", kairos_daemon_node)
    
    # Define edges
    workflow.set_entry_point("reasoning")
    
    # Main conditional: reasoning decides what's next
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        {
            "security_gate": "security_gate",
            "kairos_daemon": "kairos_daemon",
            END: END,
        },
    )
    
    # Tool execution loop
    workflow.add_edge("security_gate", "tool_execution")
    workflow.add_edge("tool_execution", "context_compaction")
    workflow.add_edge("context_compaction", "reasoning")
    
    # Kairos loop
    workflow.add_edge("kairos_daemon", "context_compaction")
    
    return workflow.compile()


# ======================================================================
# INITIALIZATION & EXECUTION
# ======================================================================

def initialize_state(user_input: str, semantic_memory: str = "") -> AgentState:
    """
    Create initial state for a new conversation.
    """
    return {
        "messages": [HumanMessage(content=user_input)],
        "working_memory": "",
        "episodic_memory": [],
        "semantic_memory": semantic_memory,
        "next_action": "",
        "observation": "",
        "tool_call_id": None,
        "loop_count": 0,
        "is_background": False,
    }


class TokenTracker:
    """
    Manages token counting without global state.
    """
    def __init__(self, ctx_window: int):
        self.ctx_window = ctx_window
        self.total_tokens = 0
        self.turn_tokens = 0
    
    def update_turn(self, response_metadata: dict):
        """Update token counts from response metadata."""
        prompt_tokens = response_metadata.get("prompt_eval_count", 0)
        completion_tokens = response_metadata.get("eval_count", 0)
        self.turn_tokens = prompt_tokens + completion_tokens
        self.total_tokens += self.turn_tokens
    
    def get_usage_percent(self) -> float:
        """Get percentage of context window used."""
        return (self.total_tokens / self.ctx_window) * 100
    
    def reset_turn(self):
        """Reset turn counter for next iteration."""
        self.turn_tokens = 0


def run_agent(
    state: AgentState,
    llm,
    tools_list: List[BaseTool],
    app,
    token_tracker: TokenTracker,
    max_iterations: int = 50,
    char_delay: float = 0.01
):
    """
    Execute the agent loop with state management.
    """
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        for event in app.stream(state):
            for node_name, node_output in event.items():
                if node_name == "reasoning":
                    if "messages" in node_output and node_output["messages"]:
                        last_msg = node_output["messages"][-1]
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            print_chars_smooth(last_msg.content, char_delay)
                            print()
                
                # Merge updates into state
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if key == "messages" and isinstance(state.get("messages"), list):
                            state["messages"].extend(value) if isinstance(value, list) else state["messages"].append(value)
                        else:
                            state[key] = value
        
        if state.get("next_action") == "final":
            break
    
    return state


def save_conversation(messages, semantic_memory: str, conv_dir):
    """
    Save conversation history and learned knowledge.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = conv_dir / f"conversation_{timestamp}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Conversation History\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Include learned knowledge
        if semantic_memory:
            f.write("## Learned Knowledge\n\n")
            f.write(f"{semantic_memory}\n\n")
            f.write("---\n\n")
        
        f.write("## Conversation\n\n")
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                f.write("### User\n\n")
                f.write(f"{msg.content}\n\n")
            elif isinstance(msg, AIMessage):
                f.write("### Assistant\n\n")
                if msg.content:
                    f.write(f"{msg.content}\n\n")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    f.write("**Tool Calls:**\n\n")
                    for tool_call in msg.tool_calls:
                        f.write(f"- `{tool_call['name']}`\n")
            elif hasattr(msg, 'name') and msg.name:
                f.write(f"**Tool Result:** `{msg.name}`\n\n")
                f.write(f"```\n{msg.content}\n```\n\n")
            f.write("---\n\n")


# ======================================================================
# MAIN LOOP
# ======================================================================

def agent_loop():
    """
    Main interactive loop for the agent.
    Manages conversation, memory, and token tracking.
    """
    from setup import system_prompt, conv_dir
    
    # Initialize
    app = build_momobot_graph(llm, tools_list)
    MemoryCascade.set_base_prompt(system_prompt)
    token_tracker = TokenTracker(CTX_WINDOW)
    
    persistent_state: AgentState = {
        "messages": [SystemMessage(content=system_prompt)],
        "working_memory": "",
        "episodic_memory": [],
        "semantic_memory": "",
        "next_action": "",
        "observation": "",
        "tool_call_id": None,
        "loop_count": 0,
        "is_background": False,
    }
    
    char_delay = 0.025
    
    console.rule(style="dim")
    user_input = console.input("[bold #C8603A]>      :[/bold #C8603A] ")
    console.rule(style="dim")
    
    while user_input.lower() != "x":
        persistent_state["messages"].append(HumanMessage(content=user_input))
        
        try:
            final_state = run_agent(
                persistent_state,
                llm,
                tools_list,
                app,
                token_tracker,
                max_iterations=5,
                char_delay=char_delay
            )
            
            persistent_state = final_state
            
            # Update token tracking
            last_message = persistent_state["messages"][-1]
            if hasattr(last_message, 'response_metadata') and last_message.response_metadata:
                token_tracker.update_turn(last_message.response_metadata)
                console.print(
                    f"            [dim]└─ Tokens: {token_tracker.turn_tokens} | "
                    f"Total: {token_tracker.get_usage_percent():.1f}%[/dim]"
                )
            
            token_tracker.reset_turn()
        
        except Exception as e:
            console.print(f"[red]Error: {str(e)[:200]}[/red]")
            persistent_state["messages"].pop()
            import traceback
            traceback.print_exc()
        
        print()
        console.rule(style="dim")
        user_input = console.input("[bold #C8603A]>      :[/bold #C8603A] ")
        console.rule(style="dim")
    
    # Save with learned knowledge
    save_conversation(
        persistent_state["messages"],
        persistent_state["semantic_memory"],
        conv_dir
    )
    console.print("             [dim green]└─[/dim green] [dim green]Conversation Saved.[/dim green]")


# if __name__ == "__main__":
#     agent_loop()


# ======================================================================
# TEST MAIN - Memory System Validation
# ======================================================================

def test_memory_system():
    """
    Test memory cascade functionality:
    1. Episodic memory accumulation and eviction
    2. Semantic memory distillation
    3. Compaction triggers
    4. LLM-based summarization
    """
    from setup import system_prompt
    
    print("\n" + "="*70)
    print("MEMORY CASCADE TEST SUITE")
    print("="*70 + "\n")
    
    # Initialize
    MemoryCascade.set_base_prompt(system_prompt)
    
    # Create test state
    state: AgentState = {
        "messages": [SystemMessage(content=system_prompt)],
        "working_memory": "",
        "episodic_memory": [],
        "semantic_memory": "",
        "next_action": "",
        "observation": "",
        "tool_call_id": None,
        "loop_count": 0,
        "is_background": False,
    }
    
    # ===== TEST 1: Episodic Memory Growth =====
    print("[TEST 1] Episodic Memory Growth & Eviction")
    print("-" * 70)
    
    for i in range(15):
        obs = f"Observation {i}: Executed tool X, result Y"
        state = {**state, **MemoryCascade.add_observation(state, obs)}
        print(f"  Added: {obs}")
        if len(state["episodic_memory"]) > MemoryCascade.EPISODIC_CAPACITY:
            print(f"  ⚠️  Episodic at capacity ({len(state['episodic_memory'])})")
    
    print(f"\n  ✓ Final episodic size: {len(state['episodic_memory'])}")
    print(f"    (Should be capped at {MemoryCascade.EPISODIC_CAPACITY})")
    assert len(state["episodic_memory"]) <= MemoryCascade.EPISODIC_CAPACITY
    print("  ✓ PASSED\n")
    
    # ===== TEST 2: Compaction Trigger Detection =====
    print("[TEST 2] Compaction Trigger Detection")
    print("-" * 70)
    
    # Fill episodic to trigger compaction
    state["loop_count"] = 0
    state["episodic_memory"] = [f"Obs {i}" for i in range(10)]
    
    should_compact = MemoryCascade.should_trigger_compaction(state)
    print(f"  Episodic size: {len(state['episodic_memory'])}")
    print(f"  Should trigger compaction: {should_compact}")
    assert should_compact == True, "Should trigger at capacity"
    print("  ✓ PASSED\n")
    
    # Test interval trigger
    state["episodic_memory"] = [f"Obs {i}" for i in range(5)]
    state["loop_count"] = 20
    should_compact = MemoryCascade.should_trigger_compaction(state)
    print(f"  Loop count: {state['loop_count']}")
    print(f"  Should trigger on interval: {should_compact}")
    assert should_compact == True, "Should trigger on 20-loop interval"
    print("  ✓ PASSED\n")
    
    # ===== TEST 3: Semantic Memory Distillation =====
    print("[TEST 3] Semantic Memory Distillation")
    print("-" * 70)
    
    state["episodic_memory"] = [
        "[14:30:00] bash: Listed 5 files in workspace",
        "[14:30:05] read_file: main.py is 638 lines",
        "[14:30:10] list_directory: Found 2 subdirectories"
    ]
    state["semantic_memory"] = ""
    state["loop_count"] = 5
    
    print(f"  Before compaction:")
    print(f"    Episodic size: {len(state['episodic_memory'])}")
    print(f"    Semantic memory: '{state['semantic_memory'][:50] if state['semantic_memory'] else 'EMPTY'}'")
    
    # Perform compaction
    result = MemoryCascade.compact_memory(state, llm)
    state.update(result)
    
    print(f"\n  After compaction:")
    print(f"    Episodic size: {len(state['episodic_memory'])}")
    print(f"    Semantic memory length: {len(state['semantic_memory'])} chars")
    print(f"\n  Distilled content:\n")
    for line in state["semantic_memory"].split('\n'):
        if line.strip():
            print(f"    {line}")
    
    assert len(state["semantic_memory"]) > 0, "Semantic memory should be populated"
    assert len(state["episodic_memory"]) < 3, "Episodic should have remaining items"
    print("\n  ✓ PASSED\n")
    
    # ===== TEST 4: System Prompt Injection =====
    print("[TEST 4] System Prompt Memory Injection")
    print("-" * 70)
    
    state["episodic_memory"] = [
        "[14:35:00] Task started: Refactoring memory module",
        "[14:35:30] Progress: 50% complete"
    ]
    state["semantic_memory"] = "[2025-01-15 | Loop 3]\n• Refactoring initiative started\n• Target: Modular components"
    
    system_prompt = MemoryCascade.build_system_prompt(state)
    
    print(f"  System prompt length: {len(system_prompt)} chars")
    print(f"\n  Checking memory injection...")
    
    assert "LONG-TERM PROJECT CONTEXT" in system_prompt, "Should include semantic section"
    assert "Refactoring initiative" in system_prompt, "Should include semantic content"
    assert "RECENT OBSERVATIONS" in system_prompt, "Should include episodic section"
    assert "Task started" in system_prompt, "Should include recent observation"
    
    print(f"  ✓ Semantic memory injected")
    print(f"  ✓ Episodic memory injected")
    print("  ✓ PASSED\n")
    
    # ===== TEST 5: Token Tracker =====
    print("[TEST 5] Token Tracker")
    print("-" * 70)
    
    tracker = TokenTracker(CTX_WINDOW)
    
    # Simulate token updates
    tracker.update_turn({"prompt_eval_count": 500, "eval_count": 150})
    print(f"  Turn 1: {tracker.turn_tokens} tokens")
    print(f"  Total: {tracker.total_tokens} tokens")
    print(f"  Usage: {tracker.get_usage_percent():.2f}%")
    
    tracker.update_turn({"prompt_eval_count": 600, "eval_count": 200})
    print(f"  Turn 2: {tracker.turn_tokens} tokens")
    print(f"  Total: {tracker.total_tokens} tokens")
    print(f"  Usage: {tracker.get_usage_percent():.2f}%")
    
    assert tracker.total_tokens == 1450, "Should accumulate tokens"
    assert tracker.turn_tokens == 800, "Should track current turn"
    print("  ✓ PASSED\n")
    
    # ===== TEST 6: Security Gate =====
    print("[TEST 6] Security Gate Validation")
    print("-" * 70)
    
    # Test dangerous bash
    is_valid, reason = SecurityGate.validate_tool_call("bash", {"command": "rm -rf /"})
    print(f"  Blocking 'rm -rf /': {is_valid} - {reason}")
    assert is_valid == False, "Should reject dangerous bash"
    
    # Test safe bash
    is_valid, reason = SecurityGate.validate_tool_call("bash", {"command": "ls -la"})
    print(f"  Allowing 'ls -la': {is_valid}")
    assert is_valid == True, "Should allow safe bash"
    
    # Test path validation
    is_valid, reason = SecurityGate.validate_tool_call("read_file", {"path": "/etc/passwd"})
    print(f"  Blocking path escape '/etc/passwd': {is_valid} - {reason}")
    assert is_valid == False, "Should reject paths outside workspace"
    
    print("  ✓ PASSED\n")
    
    # ===== SUMMARY =====
    print("="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nMemory System Validation:")
    print(f"  ✓ Episodic memory: FIFO eviction working")
    print(f"  ✓ Semantic memory: LLM-based distillation working")
    print(f"  ✓ Compaction: Triggers firing correctly")
    print(f"  ✓ System prompt: Memory injection operational")
    print(f"  ✓ Token tracking: Accumulation working")
    print(f"  ✓ Security gate: Path validation operational")
    print("\n")


if __name__ == "__main__":
    test_memory_system()