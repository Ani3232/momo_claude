# 🤖 Momobot

A local, autonomous AI agent built with LangGraph and Ollama. Momobot is designed as a thinking, acting orchestrator that manages complex tasks through a multi-tier memory system and a structured reasoning loop.

---

## ✨ Core Architecture: The Context Cascade

Unlike simple chatbots, Momobot utilizes a three-tier memory system to maintain focus and long-term project coherence:

- 🧠 **Working Memory:** Tracks the immediate state of the current turn.
- 🕒 **Episodic Memory:** A sliding window of the last 10 observations (FIFO), providing short-term session context.
- 📚 **Semantic Memory:** Long-term, distilled knowledge. The agent periodically compacts episodic observations into key facts to prevent context window bloat.

---

## ⚙️ The Agentic Loop

Momobot operates via a `StateGraph` with a rigorous execution flow:

1. **Reasoning Node:** The "Query Engine" analyzes the system prompt, memory tiers, and conversation history to decide the next action.
2. **Security Gate:** A validation layer that intercepts tool calls to block dangerous operations (e.g., `rm -rf /`).
3. **Tool Execution:** Dispatches calls to the toolset and captures observations.
4. **Context Compaction:** Periodically summarizes episodic memory into semantic memory using the LLM.
5. **Kairos Daemon:** A dedicated node for handling long-running background tasks (MVP stub).

---

## 🗂️ Project Structure

```
momobot/
├── main.py                  # Core Orchestrator, Memory Cascade, and Graph logic
├── setup.py                 # Workspace initialization and system prompt assembly
├── soul.md                  # Agent identity, personality, and core principles
├── workspace/
│   ├── user/
│   │   └── user.md          # User preferences, background, and memory log
│   ├── skills/
│   │   └── skills.md        # Behavioral rules and specialized skill indices
│   └── obsidian/           # Knowledge base and project documentation
└── conversations/           # Auto-saved session histories (Markdown)
```

---

## 🛠️ Tool Ecosystem

Momobot uses a modular toolset including:
- **Basic Tools:** File system operations (`read`, `write`, `list`), Web Search.
- **Task State Tools:** Management of complex goals, step tracking, and replanning.
- **Subagent Tool:** Delegation of deep research or long-chain reasoning to specialized sub-instances.
- **Clarification Tool:** Proactive questioning when user requirements are ambiguous.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally.

### 2. Install dependencies
```bash
pip install langchain langchain-community langchain-ollama langgraph termcolor langchain-experimental rich
```

### 3. Execution
Update the `agent_model` in `main.py` to your preferred local model (e.g., `gemma4:31b-cloud`) and run:
```bash
python main.py
```

---

## 💡 Customization
- **Personality:** Edit `soul.md` to refine how the agent thinks and speaks.
- **User Context:** Update `workspace/user/user.md` to provide background about yourself.
- **Capabilities:** Add new skill documents to `workspace/skills/` and register them in `skills.md`.

---

## 📄 License
MIT — build, break, and improve.
