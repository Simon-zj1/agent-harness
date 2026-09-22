"""OpenAI-compatible chat completions (DeepSeek, local Ollama, LM Studio, ...)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ..config import ProviderConfig, secret
from ..errors import ProviderError
from .base import ChatResponse, ToolCall

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    """A thin, dependency-free client: enough for chat + tool calling."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name
        self.base_url = config.base_url.rstrip("/")

    def available(self) -> tuple[bool, str]:
        if not self.base_url:
            return False, "base_url is empty"
        env_name = self.config.api_key_env
        if env_name and not secret(env_name):
            return False, f"{env_name} is not set (env or ~/.codex/.env)"
        if "127.0.0.1" in self.base_url or "localhost" in self.base_url:
            import socket
            import urllib.parse

            parsed = urllib.parse.urlparse(self.base_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            with socket.socket() as sock:
                sock.settimeout(1.5)
                if sock.connect_ex((host, port)) != 0:
                    return False, f"no server listening on {host}:{port}"
        return True, "configured"

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
        limit = max_tokens or self.config.max_output_tokens
        if limit:
            payload["max_tokens"] = limit

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                body = self._post("/chat/completions", payload)
                return self._parse(body)
            except ProviderError as exc:
                last_error = exc
                status = getattr(exc, "status", 0)
                if status not in RETRY_STATUS or attempt == self.config.max_retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = ProviderError(f"network failure calling {self.base_url}: {exc}")
                if attempt == self.config.max_retries:
                    raise last_error
                time.sleep(min(2 ** attempt, 8))
        raise ProviderError(f"provider {self.name} failed: {last_error}")

    # -- internals ---------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "agent-harness/0.1",
            **self.config.extra_headers,
        }
        key = secret(self.config.api_key_env) if self.config.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            error = ProviderError(f"HTTP {exc.code} from {url}: {detail}")
            error.status = exc.code  # type: ignore[attr-defined]
            raise error from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"cannot reach {url}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderError(f"non-JSON response from {url}") from exc

    def _parse(self, body: dict) -> ChatResponse:
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"provider returned no choices: {str(body)[:300]}")
        choice = choices[0]
        msg = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for call in msg.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_args = function.get("arguments") or "{}"
            try:
                parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed = {"__invalid_json__": raw_args}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{len(tool_calls)}",
                    name=function.get("name", ""),
                    arguments=parsed if isinstance(parsed, dict) else {"value": parsed},
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                )
            )
        usage = body.get("usage") or {}
        response = ChatResponse(
            text=msg.get("content") or "",
            tool_calls=tool_calls,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=body.get("model", self.config.model),
            finish_reason=choice.get("finish_reason", ""),
            raw=body,
        )
        response.cost_usd = self._cost(response)
        return response

    def _cost(self, response: ChatResponse) -> float | None:
        if self.config.input_per_mtok is None or self.config.output_per_mtok is None:
            return None
        return (
            response.tokens_in / 1_000_000 * self.config.input_per_mtok
            + response.tokens_out / 1_000_000 * self.config.output_per_mtok
        )
