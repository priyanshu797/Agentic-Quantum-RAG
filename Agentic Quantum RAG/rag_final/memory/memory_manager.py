import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Deque, Optional

@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Scratchpad:
    thoughts:   List[str]            = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0

    def reset(self):
        self.thoughts.clear(); self.tool_calls.clear(); self.iterations = 0

    def think(self, t: str):
        self.thoughts.append(t)

    def record(self, name: str, inp: Any, out: Any):
        self.tool_calls.append({"tool": name, "input": inp, "output": out})

class Session:
    def __init__(self, sid: str):
        self.session_id  = sid
        self.messages:   Deque[Message] = deque(maxlen=20)
        self.scratchpad  = Scratchpad()
        self.created_at  = time.time()
        self.last_active = time.time()

    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.messages.append(Message(role=role, content=content, metadata=metadata or {}))
        self.last_active = time.time()

    def history(self, n: int = 6) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in list(self.messages)[-n:]]

    def history_str(self, n: int = 4) -> str:
        return "\n".join(
            ("You" if m.role == "user" else "AI") + ": " + m.content
            for m in list(self.messages)[-n:]
        )

    def full_context(self) -> str:
        return "\n".join(
            ("You" if m.role == "user" else "AI") + ": " + m.content
            for m in list(self.messages)
        )

class MemoryManager:
    TTL = 3600
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, sid: str) -> Session:
        self._evict()
        if sid not in self._sessions:
            self._sessions[sid] = Session(sid)
        return self._sessions[sid]

    def get(self, sid: str) -> Optional[Session]:
        return self._sessions.get(sid)

    def _evict(self):
        now = time.time()
        for sid in [s for s, m in self._sessions.items() if now - m.last_active > self.TTL]:
            del self._sessions[sid]

memory_manager = MemoryManager()
