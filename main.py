# ======================================================================||
# IMPORTS                                                               ||
# ======================================================================||
import operator
import sys
import time
from typing import Annotated, TypedDict, Union, List, Optional
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from rich.console import Console
from setup import workspace

from basic_tools import base_tools
from subagent_tool import subagent_tool
from clarification_tool import ask_clarifying_questions_tool
from task_state_tool import task_state_tools
import sys
import time
console = Console()

scaler = 1.01

def print_chars_smooth(text: str, char_delay: float):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if char_delay > 0:
            time.sleep(char_delay * scaler)
# ======================================================================|
# agent_model = "minimax-m2.7:cloud"  # 204800
agent_model = "gemma4:31b-cloud"
CTX_WINDOW =  256000
total_tokens_used = 0
turn_tokens = 0

llm = ChatOllama(
    model = agent_model,
    base_url="http://localhost:11434",
    num_ctx=CTX_WINDOW,
    stream=False
)
tools_list = base_tools + task_state_tools + [subagent_tool ,ask_clarifying_questions_tool]
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
# ORCHESTRATOR: Memory Cascade Logic
# ======================================================================

class MemoryCascade:
    """
    Manages the three-tier memory system.
    - Injects relevant context into LLM prompts
    - Compacts episodic → semantic on overflow
    - Maintains budget constraints
    """
    
    EPISODIC_CAPACITY = 10  # Keep last 10 observations
    SEMANTIC_MAX_TOKENS = 500  # Semantic memory size limit
    BASE_SYSTEM_PROMPT = None  # Will be set from setup
    
    @staticmethod
    def set_base_prompt(prompt: str):
        """Set the base system prompt from setup."""
        MemoryCascade.BASE_SYSTEM_PROMPT = prompt
    
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
        episodic_str = ""
        if state["episodic_memory"]:
            episodic_str = "Recent observations:\n" + "\n".join(
                f"  • {obs}" for obs in state["episodic_memory"][-5:]  # Last 5
            )
        
        semantic_str = ""
        if state["semantic_memory"]:
            semantic_str = f"Project context:\n{state['semantic_memory']}"
        
        # Combine base prompt with memory context
        memory_context = ""
        if semantic_str:
            memory_context += semantic_str + "\n\n"
        if episodic_str:
            memory_context += episodic_str + "\n"
        
        system = f"""{base}

        {memory_context}"""
        
        return system
    
    @staticmethod
    def add_observation(state: AgentState, observation: str, tool_call_id: str) -> dict:
        """
        After tool execution, integrate observation into episodic memory.
        Trigger compaction if episodic memory exceeds capacity.
        """
        episodic = state["episodic_memory"]
        timestamp = datetime.now().strftime("%H:%M:%S")
        episodic.append(f"[{timestamp}] {observation[:100]}")  # Truncate long observations
        
        # Keep only recent observations
        if len(episodic) > MemoryCascade.EPISODIC_CAPACITY:
            # In production: distill oldest into semantic_memory
            # For now, simple FIFO eviction
            episodic = episodic[-MemoryCascade.EPISODIC_CAPACITY:]
        
        return {
            "episodic_memory": episodic,
            "observation": observation,
            "tool_call_id": tool_call_id,
        }
    
    @staticmethod
    def should_compact(state: AgentState) -> bool:
        """Trigger semantic memory compaction if episodic is getting full."""
        return len(state["episodic_memory"]) >= MemoryCascade.EPISODIC_CAPACITY


# ======================================================================
# NODE 1: REASONING NODE - The Query Engine
# ======================================================================

def reasoning_node(state: AgentState, llm, tools_list: List[BaseTool]):
    """
    Core orchestrator: LLM-driven reasoning that decides what to do next.
    
    This is the "Query Engine" from the spec:
    - Analyzes current state + memory + conversation
    - Decides: use tool, run kairos daemon, or provide final answer
    - Uses LLM's native tool_calls mechanism (no keyword matching)
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
    working_mem = f"LLM response: {response.content[:50] if response.content else 'tool call'}"
    
    return {
        "messages": [response],
        "next_action": next_action,
        "working_memory": working_mem,
        "loop_count": state.get("loop_count", 0) + 1,
    }


# ======================================================================
# NODE 2: SECURITY & PERMISSIONS GATE
# ======================================================================

def security_gate_node(state: AgentState) -> dict:
    """
    Validate tool calls before execution.
    
    In production, this would:
    - Check permissions (user can execute this tool?)
    - Validate arguments (safe paths, no dangerous operations?)
    - Log audit trail
    - Implement rate limiting
    
    For MVP: validate and reject dangerous calls, pass through safe ones.
    """
    
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # No tool call to validate; pass through
        return {}
    
    tool_calls = last_message.tool_calls
    rejected_calls = []
    
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        
        # Example validation: prevent dangerous file operations
        if tool_name == "bash" and any(
            dangerous in str(args).lower() 
            for dangerous in ["rm -rf /", "sudo", "dd if=/dev/"]
        ):
            # REJECT: too dangerous
            rejected_calls.append({
                "call": call,
                "reason": f"Dangerous bash command rejected: {tool_name}"
            })
    
    # If any calls were rejected, inject rejection messages
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
    
    # All calls passed validation — pass through (no changes)
    return {}


# ======================================================================
# NODE 3: TOOL EXECUTION NODE
# ======================================================================

def tool_execution_node(state: AgentState, tools_list: List[BaseTool]) -> dict:
    """
    Execute the tool(s) that the LLM selected.
    Uses LangChain's ToolNode for real tool dispatch.
    """
    
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {}
    
    # Create a ToolNode and invoke it
    tool_node = ToolNode(tools=tools_list)
    
    try:
        # ToolNode.invoke() expects messages as input
        # It finds tool_calls in the last AIMessage and executes them
        tool_results = tool_node.invoke({"messages": state["messages"]})
        
        # tool_results["messages"] contains ToolMessage objects with real outputs
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
    updated_episodic = state["episodic_memory"].copy()
    for msg in tool_messages:
        timestamp = datetime.now().strftime("%H:%M:%S")
        updated_episodic.append(f"[{timestamp}] {msg.name}: {msg.content[:80]}")
    
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
    
    Uses LLM to summarize, not just string concat.
    
    Triggered when:
    - Episodic memory exceeds capacity
    - Loop count hits compaction interval (e.g., every 20 loops)
    - Conversation changes major topics
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
    
    # Use LLM to summarize episodic memory
    episodic_str = "\n".join(episodic[-5:])  # Last 5 observations
    
    try:
        summary_prompt = f"""Summarize these recent observations into 2-3 key facts for long-term memory:

{episodic_str}

Summary:"""
        
        summary_response = llm.invoke(summary_prompt)
        summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        
    except Exception as e:
        # Fallback to simple concatenation if LLM fails
        summary = "; ".join(episodic[-3:])
    
    # Append summary to semantic memory
    new_semantic = state["semantic_memory"] + f"\n[Loop {loop_count}] {summary}"
    
    # Trim semantic memory if too large
    if len(new_semantic) > MemoryCascade.SEMANTIC_MAX_TOKENS:
        new_semantic = new_semantic[-MemoryCascade.SEMANTIC_MAX_TOKENS:]
    
    return {
        "semantic_memory": new_semantic,
        "episodic_memory": episodic[-3:],  # Keep only very recent
    }


# ======================================================================
# NODE 5: KAIROS DAEMON (Background Tasks)
# ======================================================================

def kairos_daemon_node(state: AgentState) -> dict:
    """
    Long-running autonomous task handler.
    
    Scenario: User says "Monitor the build logs in the background
    and notify me when it completes."
    
    Kairos breaks from the request-response cycle and runs until
    a goal condition is met.
    
    TODO: Real implementation uses threading.Thread or asyncio.Task
    """
    
    # For MVP: stub — just acknowledge the background task
    # Real implementation would:
    # - Fork a background thread with its own reason-act-observe loop
    # - Poll for completion condition
    # - Feed observations back into episodic memory
    # - Resume main loop when done
    
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
    - "tools": Execute a tool call
    - "kairos": Start autonomous daemon
    - END: Provide final response to user
    """
    next_action = state.get("next_action", "final")
    
    if next_action == "tool":
        return "security_gate"
    elif next_action == "kairos":
        return "kairos_daemon"
    else:
        return END


def route_after_tool(state: AgentState) -> str:
    """
    After tool execution, return to reasoning.
    """
    return "context_compaction"


# ======================================================================
# GRAPH COMPILATION
# ======================================================================

def build_momobot_graph(llm, tools_list: List[BaseTool]):
    """
    Assemble the complete orchestrator graph.
    
    Flow:
    reasoning → {tool path or final}
    tool path: reasoning → security_gate → tool_execution → context_compaction → reasoning
    kairos path: reasoning → kairos_daemon → context_compaction → reasoning
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
    # After kairos compaction, could return to reasoning or end
    # For now: goes back to reasoning
    # workflow.add_conditional_edges("context_compaction_kairos", ...)
    
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


def run_agent(state: AgentState, llm, tools_list: List[BaseTool], app, max_iterations: int = 50, char_delay: float = 0.01):
    from setup import system_prompt
    
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

def count_tokens(messages):
    return int(turn_tokens)

def save_conversation(messages, conv_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = conv_dir / f"conversation_{timestamp}.md"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Conversation History\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        for msg in messages:
            if isinstance(msg, HumanMessage):
                f.write("##  User\n\n")
                f.write(f"{msg.content}\n\n")
            elif isinstance(msg, AIMessage):
                f.write("##  Assistant\n\n")
                if msg.content:
                    f.write(f"{msg.content}\n\n")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    f.write("### Tool Calls\n\n")
                    for tool_call in msg.tool_calls:
                        f.write(f"**Tool:** `{tool_call['name']}`\n\n")
                        if 'args' in tool_call:
                            f.write("**Arguments:**\n```json\n")
                            f.write(f"{tool_call['args']}\n")
                            f.write("```\n\n")
            elif hasattr(msg, 'name') and msg.name:
                f.write(f"### Tool Result: {msg.name}\n\n")
                f.write("```\n")
                f.write(f"{msg.content}\n")
                f.write("```\n\n")
            f.write("---\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- **Total messages:** {len(messages)}\n")
        f.write(f"- **Tokens (approx):** {count_tokens(messages)}\n")
        f.write(f"- **File saved:** {filename.name}\n")


# ======================================================================||
# MAIN LOOP                                                             ||
# ======================================================================||
def agent_loop():
    global turn_tokens, total_tokens_used

    from setup import system_prompt, conv_dir

    app = build_momobot_graph(llm, tools_list)

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

    while user_input != "x":
        persistent_state["messages"].append(HumanMessage(content=user_input))

        try:
            final_state = run_agent(persistent_state, llm, tools_list, app, max_iterations=5, char_delay=char_delay)
            
            persistent_state = final_state
            
            last_message = persistent_state["messages"][-1]
            if hasattr(last_message, 'response_metadata') and last_message.response_metadata:
                meta = last_message.response_metadata
                turn_tokens = (meta.get("prompt_eval_count") or 0) + (meta.get("eval_count") or 0)
                total_tokens_used = (turn_tokens/CTX_WINDOW)*100
                console.print(f"            [dim]└─ Tokens: {turn_tokens} | Total: {total_tokens_used}%[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {str(e)[:200]}[/red]")
            persistent_state["messages"].pop()
            import traceback
            traceback.print_exc()

        print()
        console.rule(style="dim")
        user_input = console.input("[bold #C8603A]>      :[/bold #C8603A] ")
        console.rule(style="dim")

    save_conversation(persistent_state["messages"], conv_dir)
    console.print("             [dim green]└─[/dim green] [dim green]Conversation Saved.[/dim green]")


if __name__ == "__main__":
    agent_loop()