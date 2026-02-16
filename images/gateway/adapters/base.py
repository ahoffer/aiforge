"""Base model adapter with passthrough defaults.

Every method returns its input unchanged. Models with native tool calling
need no overrides. Models that emit tool calls as content text override
the normalize methods to recover structured tool_calls from free text.
"""

import logging

from langchain_core.messages import AIMessage


log = logging.getLogger(__name__)


class ModelAdapter:
    """Strategy interface for model-specific behavior.

    Proxy path uses normalize_tool_calls and inject_tool_guidance.
    Agent path uses normalize_ai_message and llm_kwargs.
    Both paths share _filter_hallucinated for safety.
    """

    def normalize_tool_calls(self, ollama_msg: dict,
                             valid_names: set[str]) -> dict:
        """Post-process an Ollama response message on the proxy path.

        Subclasses override this to recover tool_calls from content text
        when the model does not use structured tool calling natively.
        Base implementation filters hallucinated names only.
        """
        tool_calls = ollama_msg.get("tool_calls")
        if tool_calls:
            ollama_msg = dict(ollama_msg)
            ollama_msg["tool_calls"] = self._filter_hallucinated(
                tool_calls, valid_names)
        return ollama_msg

    def normalize_ai_message(self, message: AIMessage) -> AIMessage:
        """Post-process an AIMessage from ChatOllama on the agent path.

        Subclasses override this to recover tool_calls from content text.
        Base implementation is passthrough.
        """
        return message

    def inject_tool_guidance(self, ollama_messages: list[dict],
                             tools: list[dict] | None) -> list[dict]:
        """Prepend steering prompts to help the model select tools.

        Base implementation is passthrough. Models that struggle with
        tool selection override this with model-specific guidance.
        """
        return ollama_messages

    def llm_kwargs(self) -> dict:
        """Extra keyword arguments for the ChatOllama constructor.

        Override to pass model-specific flags like think=True for
        models that support extended thinking.
        """
        return {}

    def _filter_hallucinated(self, tool_calls: list[dict],
                             valid_names: set[str]) -> list[dict]:
        """Reject tool_calls whose names the client never offered.

        Catches the common failure where pretraining priors overwhelm
        injected schemas and the model invents plausible-looking names.
        """
        if not valid_names or not tool_calls:
            return tool_calls
        kept = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            if name in valid_names:
                kept.append(tc)
            else:
                log.warning("filtered hallucinated tool_call name=%r", name)
        return kept
