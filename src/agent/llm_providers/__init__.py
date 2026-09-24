"""
G8 -- real LLM provider adapters.

Each module here exposes exactly one factory, make_<provider>_call_llm(...),
returning a plain closure matching src.agent.reasoning.CallLLM
(Callable[[str], Awaitable[str]]) -- nothing more. src.agent.reasoning,
src.agent.reasoning_validator, and src.agent.case_output never import from
this package; a provider is wired in only by whoever calls
case_output.investigate_and_assemble(..., call_llm=make_gemini_call_llm(...)),
exactly the same way a test's fake call_llm is wired in. This package is
also isolated from src.agent.workflow, src.policy.g2_policy,
src.investigation.patterns, and any TigerGraph code -- a provider adapter
has no access to any of them and no way to reach them.

Importing this package does not require the underlying provider SDK to be
installed: each provider module imports its SDK lazily, inside its factory
function, so the rest of this project's test suite never depends on it.
"""
from __future__ import annotations
