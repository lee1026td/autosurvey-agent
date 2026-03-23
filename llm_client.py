"""LLM Client for llama-server with OpenAI-compatible API.

Supports both chat completions and embeddings via /v1/chat/completions
and /v1/embeddings endpoints.
"""

import json
import re
import time
from typing import Any

from openai import OpenAI


class LLMClient:
    """OpenAI-compatible client for llama-server (LLM + Embeddings)."""

    SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 1.5,
    }
    EXTRA_SAMPLING_PARAMS = {
        "top_k": 20,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        embed_host: str | None = None,
        embed_port: int | None = None,
        stream_summary: bool = False,
        stream_reasoning: bool = False,
        trace_latency: bool = True,
    ):
        # Chat completions client
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="sk-no-key-required")
        models = self.client.models.list().data
        if not models:
            raise RuntimeError("No models available from llama-server.")
        self.model = models[0].id

        # Embedding client (separate server if specified)
        if embed_host and embed_port:
            self.embed_client = OpenAI(
                base_url=f"http://{embed_host}:{embed_port}/v1",
                api_key="sk-no-key-required",
            )
            embed_models = self.embed_client.models.list().data
            if not embed_models:
                raise RuntimeError("No models available from embedding server.")
            self.embed_model = embed_models[0].id
        else:
            self.embed_client = self.client
            self.embed_model = self.model

        self.stream_summary = stream_summary
        self.stream_reasoning = stream_reasoning
        self.trace_latency = trace_latency

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        reasoning: bool = False,
        *,
        stream: bool = False,
        stream_label: str = "",
    ) -> str:
        """Generate a chat completion response."""
        think_tag = "/think" if reasoning else "/no_think"
        start = time.perf_counter()
        extra_body = {
            **self.EXTRA_SAMPLING_PARAMS,
            "enable_thinking": reasoning,
            "enable_reasoning": reasoning,
            "chat_template_kwargs": {
                "enable_thinking": reasoning,
                "enable_reasoning": reasoning,
            },
        }

        if stream:
            chunks = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{think_tag}\n{user_prompt}"},
                ],
                stream=True,
                **self.SAMPLING_PARAMS,
                extra_body=extra_body,
            )
            text = self._consume_stream(chunks, stream_label=stream_label)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{think_tag}\n{user_prompt}"},
                ],
                **self.SAMPLING_PARAMS,
                extra_body=extra_body,
            )
            text = response.choices[0].message.content or ""

        if self.trace_latency:
            elapsed = time.perf_counter() - start
            tag = f" [{stream_label}]" if stream_label else ""
            mode = "think" if reasoning else "no_think"
            print(f"[llm]{tag} mode={mode} elapsed={elapsed:.2f}s")

        return text.strip()

    def ask_json(
        self,
        system_prompt: str,
        user_prompt: str,
        reasoning: bool = False,
        max_retries: int = 2,
        *,
        stream: bool = False,
        stream_label: str = "",
    ) -> dict[str, Any]:
        """Generate a chat completion and parse as JSON."""
        last_error = None
        for attempt in range(max_retries + 1):
            text = self.ask(
                system_prompt,
                user_prompt,
                reasoning=reasoning,
                stream=stream,
                stream_label=stream_label,
            )
            try:
                return self._extract_json(text)
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries:
                    continue
        raise last_error  # type: ignore

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text.

        Args:
            text: Input text to embed
        """
        start = time.perf_counter()

        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding

        if self.trace_latency:
            elapsed = time.perf_counter() - start
            dim = len(embedding)
            print(f"[embed] elapsed={elapsed:.2f}s dim={dim}")

        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
        """
        if not texts:
            return []

        start = time.perf_counter()

        # Keep embedding batches small because local llama-server embedding backends
        # can fail when a single request contains too many long inputs.
        batch_size = 8
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.embed_client.embeddings.create(
                    model=self.embed_model,
                    input=batch,
                    encoding_format="float",
                )
                batch_embeddings = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([e.embedding for e in batch_embeddings])
            except Exception:
                # Fallback to one-by-one embedding so a single oversized request does
                # not fail the entire indexing pass.
                for text in batch:
                    response = self.embed_client.embeddings.create(
                        model=self.embed_model,
                        input=text,
                        encoding_format="float",
                    )
                    all_embeddings.append(response.data[0].embedding)

        if self.trace_latency:
            elapsed = time.perf_counter() - start
            print(f"[embed_batch] elapsed={elapsed:.2f}s count={len(texts)}")

        return all_embeddings



    def _consume_stream(self, stream_iter: Any, stream_label: str = "") -> str:
        """Consume streaming response and return combined text."""
        collected_text: list[str] = []
        reasoning_chunks: list[str] = []

        prefix = f"[{stream_label}] " if stream_label else ""
        print(f"[stream] {prefix}start")

        for chunk in stream_iter:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                collected_text.append(content)
                print(content, end="", flush=True)

            if self.stream_reasoning:
                reasoning_text = getattr(delta, "reasoning_content", None)
                if reasoning_text:
                    reasoning_chunks.append(reasoning_text)
                    print(reasoning_text, end="", flush=True)

        print("\n[stream] end")

        combined = "".join(collected_text)
        if self.stream_reasoning and reasoning_chunks and "<think>" not in combined:
            combined = f"<think>{''.join(reasoning_chunks)}</think>\n{combined}"
        return combined

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Extract JSON object from text that may contain thinking tags or markdown."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        start_idx = text.find("{")
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        depth = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if depth != 0:
            raise json.JSONDecodeError("Unbalanced braces", text, start_idx)

        candidate = text[start_idx : end_idx + 1]
        return json.loads(candidate)
