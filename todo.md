# Momobot Development Roadmap

## 🚀 Core Architecture Refinements

### [X] Kairos Daemon Implementation (Not now - Ani prefers manual visibility)
- [ ] Replace current stub with a real background execution system.
- [ ] Implement `threading.Thread` or `asyncio.Task` to allow non-blocking autonomous loops.
- [ ] Create a mechanism for the daemon to feed observations back into the main episodic memory.
- [ ] Define goal-condition triggers to signal the daemon's completion.

### [ ] Memory Cascade & Compaction
- [ ] Upgrade `MemoryCascade.add_observation` from simple FIFO eviction to intelligent distillation.
- [ ] Implement a "summary-of-summaries" logic for semantic memory to avoid crude string slicing.
- [ ] Refine the trigger for `context_compaction_node` to include topic-shift detection.

### [ ] Security & Permissions Gate
- [ ] Transition from hardcoded keyword blocking to a robust validation system.
- [ ] Implement an audit trail for all tool executions.
- [ ] Add argument validation to ensure paths stay within the workspace directory.
- [ ] Build a permissions layer to control tool access.

## 🛠 Technical Debt & Refactoring
- [ ] Refactor monolithic files (`main.py`, `task_state_tool.py`, `subagent_tool.py`) into modular components.
- [ ] Remove global state in `main.py` (specifically token tracking) in favor of a state management class.
- [ ] Decouple `subagent_tool.py` by moving PDF and graph logic into separate utility files.
