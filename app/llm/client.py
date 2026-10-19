"""
Unified LLM client adapter.
Wraps Anthropic, OpenAI, Google Gemini, and Groq behind a single interface.

Usage:
    client = build_llm_client(settings)
    response = client.chat(model, system, tools, messages, max_tokens, temperature)
    for tu in response.tool_uses:   # tu.id, tu.name, tu.args
        ...
    print(response.text)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger()


# ── Normalised response ───────────────────────────────────────────────────────

@dataclass
class ToolUse:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    tool_uses: list[ToolUse] = field(default_factory=list)
    text: str = ""
    stop_reason: str = ""
    raw: Any = None  # original provider response


# ── Provider adapters ─────────────────────────────────────────────────────────

class AnthropicAdapter:
    """Wraps anthropic.Anthropic client."""

    def __init__(self, client):
        self._client = client

    def chat(self, model, system, tools, messages, max_tokens=2048, temperature=0) -> LLMResponse:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,          # already in Anthropic format
            messages=messages,
            temperature=temperature,
        )
        tool_uses = []
        text = ""
        for block in resp.content:
            if block.type == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, args=block.input))
            elif block.type == "text":
                text += block.text
        return LLMResponse(tool_uses=tool_uses, text=text, stop_reason=resp.stop_reason, raw=resp)

    def make_tool_result_message(self, tool_use_id: str, content: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
        }

    def make_assistant_message(self, raw_response) -> dict:
        return {"role": "assistant", "content": raw_response.content}


class OpenAIAdapter:
    """Wraps openai.OpenAI or groq.Groq (both are OpenAI-compatible)."""

    def __init__(self, client):
        self._client = client

    def _to_openai_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool format → OpenAI tool format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    def _to_openai_messages(self, system: str, messages: list[dict]) -> list[dict]:
        """Prepend system as a message."""
        out = [{"role": "system", "content": system}]
        for m in messages:
            # Convert Anthropic tool_result format to OpenAI tool message
            if isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        # skip — covered by assistant message
                        pass
                    elif isinstance(block, dict) and block.get("type") == "text":
                        out.append({"role": "assistant", "content": block["text"]})
            else:
                out.append(m)
        return out

    def chat(self, model, system, tools, messages, max_tokens=2048, temperature=0) -> LLMResponse:
        oai_tools = self._to_openai_tools(tools)
        oai_messages = self._to_openai_messages(system, messages)

        resp = self._client.chat.completions.create(
            model=model,
            messages=oai_messages,
            tools=oai_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg = choice.message

        tool_uses = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_uses.append(ToolUse(id=tc.id, name=tc.function.name, args=args))

        return LLMResponse(
            tool_uses=tool_uses,
            text=msg.content or "",
            stop_reason=choice.finish_reason,
            raw=resp,
        )

    def make_tool_result_message(self, tool_use_id: str, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_use_id, "content": content}

    def make_assistant_message(self, raw_response) -> dict:
        choice = raw_response.choices[0]
        msg = choice.message
        # Reconstruct assistant message with tool_calls if present
        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return assistant


class GeminiAdapter:
    """Wraps google.generativeai GenerativeModel."""

    def __init__(self, client):
        # client is a genai.GenerativeModel instance
        self._client = client
        self._model_name = None

    def _to_gemini_tools(self, tools: list[dict]):
        import google.generativeai as genai
        declarations = []
        for t in tools:
            schema = t.get("input_schema", {})
            declarations.append(genai.protos.FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(type=self._gemini_type(v.get("type", "string")),
                                               description=v.get("description", ""),
                                               enum=v.get("enum", []))
                        for k, v in schema.get("properties", {}).items()
                    },
                    required=schema.get("required", []),
                ),
            ))
        return [genai.protos.Tool(function_declarations=declarations)]

    @staticmethod
    def _gemini_type(t: str):
        import google.generativeai as genai
        return {
            "string": genai.protos.Type.STRING,
            "integer": genai.protos.Type.INTEGER,
            "boolean": genai.protos.Type.BOOLEAN,
            "number": genai.protos.Type.NUMBER,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT,
        }.get(t, genai.protos.Type.STRING)

    def chat(self, model, system, tools, messages, max_tokens=2048, temperature=0) -> LLMResponse:
        import google.generativeai as genai

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temperature),
        )

        # Convert messages to Gemini history format
        history = []
        last_user = None
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            content = m["content"] if isinstance(m["content"], str) else str(m.get("content", ""))
            history.append({"role": role, "parts": [content]})

        chat = gemini_model.start_chat(history=history[:-1] if history else [])
        last_msg = history[-1]["parts"][0] if history else ""

        gemini_tools = self._to_gemini_tools(tools)
        resp = chat.send_message(last_msg, tools=gemini_tools)

        tool_uses = []
        text = ""
        for part in resp.parts:
            if part.function_call.name:
                fc = part.function_call
                args = dict(fc.args)
                tool_uses.append(ToolUse(id=f"fc_{fc.name}", name=fc.name, args=args))
            elif part.text:
                text += part.text

        return LLMResponse(tool_uses=tool_uses, text=text, stop_reason="end_turn", raw=resp)

    def make_tool_result_message(self, tool_use_id: str, content: str) -> dict:
        # Gemini uses function_response
        return {"role": "user", "content": f"[Tool result: {content}]"}

    def make_assistant_message(self, raw_response) -> dict:
        parts = []
        for part in raw_response.parts:
            if part.text:
                parts.append(part.text)
        return {"role": "assistant", "content": " ".join(parts)}


# ── Factory ───────────────────────────────────────────────────────────────────

def build_llm_client(settings) -> AnthropicAdapter | OpenAIAdapter | GeminiAdapter:
    provider = (settings.llm_provider or "anthropic").lower()

    if provider == "anthropic":
        import anthropic
        raw = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        log.info("llm_provider_loaded", provider="anthropic")
        return AnthropicAdapter(raw)

    if provider == "openai":
        import openai
        raw = openai.OpenAI(api_key=settings.openai_api_key)
        log.info("llm_provider_loaded", provider="openai")
        return OpenAIAdapter(raw)

    if provider == "groq":
        from groq import Groq
        raw = Groq(api_key=settings.groq_api_key)
        log.info("llm_provider_loaded", provider="groq")
        return OpenAIAdapter(raw)  # Groq is OpenAI-compatible

    if provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        log.info("llm_provider_loaded", provider="google")
        return GeminiAdapter(None)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose: anthropic, openai, groq, google")
