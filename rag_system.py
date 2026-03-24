"""RAG (Retrieval-Augmented Generation) system for research Q&A."""

import json
import re
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from vector_store import VectorStore


RAG_SYSTEM_PROMPT = """You are a helpful research assistant. Answer questions based on the provided research documents.

Rules:
- Use ONLY information from the provided documents
- Cite document IDs when referencing specific information following this format: [Document_path/Document doc_id]
- If the documents don't contain relevant information, say so clearly
- Be concise but comprehensive
"""

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that captures the full context.

CONVERSATION HISTORY:
{history}

FOLLOW-UP QUESTION: {question}

Rewrite the follow-up question as a standalone search query. Output ONLY the rewritten query, nothing else."""

RAG_USER_PROMPT_TEMPLATE = """Based on the following research documents, answer the user's question.

DOCUMENTS:
{context}

RECENT CONVERSATION:
{history}

USER QUESTION: {question}

Provide a clear, well-structured answer based on the documents above."""


class RAGSystem:
    """Retrieval-Augmented Generation system for research Q&A."""

    def __init__(
        self,
        llm: LLMClient,
        vector_store: VectorStore,
        n_results: int = 5,
        max_context_chars: int = 12000,
        max_embed_chars: int = 900,
        chunk_overlap_chars: int = 120,
        max_history_turns: int = 3,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.n_results = n_results
        self.max_context_chars = max_context_chars
        self.max_embed_chars = max_embed_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.max_history_turns = max_history_turns
        self.chat_history: list[tuple[str, str]] = []

    def _format_recent_history(self) -> str:
        if not self.chat_history:
            return "(No previous conversation)"

        recent = self.chat_history[-self.max_history_turns :]
        parts: list[str] = []
        for i, (user_q, assistant_a) in enumerate(recent, start=1):
            parts.append(f"Turn {i} User: {user_q}")
            parts.append(f"Turn {i} Assistant: {assistant_a}")
        return "\n".join(parts)

    def _normalize_for_embedding(self, text: str) -> str:
        text = text.replace("\r\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        text = self._normalize_for_embedding(text)
        if not text:
            return []
        if len(text) <= self.max_embed_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.max_embed_chars, text_len)
            if end < text_len:
                split_at = max(
                    text.rfind("\n\n", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind(" ", start, end),
                )
                if split_at > start + (self.max_embed_chars // 3):
                    end = split_at + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break
            start = max(end - self.chunk_overlap_chars, start + 1)

        return chunks

    def _append_chunked_document(
        self,
        *,
        base_doc_id: str,
        content: str,
        metadata: dict[str, Any],
        doc_ids: list[str],
        contents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        chunks = self._chunk_text(content)
        for chunk_index, chunk in enumerate(chunks):
            chunk_doc_id = base_doc_id if len(chunks) == 1 else f"{base_doc_id}:chunk_{chunk_index:03d}"
            chunk_metadata = dict(metadata)
            chunk_metadata.update(
                {
                    "parent_doc_id": base_doc_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
            )
            doc_ids.append(chunk_doc_id)
            contents.append(chunk)
            metadatas.append(chunk_metadata)

    def index_documents(
        self,
        summaries_dir: Path,
        index_path: Path | None = None,
    ) -> int:
        metadata_map: dict[str, dict[str, str]] = {}
        if index_path and index_path.exists():
            try:
                index_data = json.loads(index_path.read_text(encoding="utf-8"))
                for record in index_data.get("records", []):
                    metadata_map[record["doc_id"]] = {
                        "title": record.get("title", ""),
                        "url": record.get("url", ""),
                        "domain": record.get("domain", ""),
                        "search_query": record.get("search_query", ""),
                    }
            except Exception as e:
                print(f"[rag] Warning: Could not load index.json: {e}")

        summary_files = sorted(summaries_dir.glob("doc_*.md"))
        if not summary_files:
            print("[rag] No summary files found to index")
            return 0

        doc_ids: list[str] = []
        contents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for summary_file in summary_files:
            if "_error" in summary_file.stem:
                continue

            doc_id = summary_file.stem.replace("doc_", "")
            content = summary_file.read_text(encoding="utf-8")

            if "Duplicate of:" in content:
                continue

            self._append_chunked_document(
                base_doc_id=doc_id,
                content=content,
                metadata=metadata_map.get(doc_id, {}),
                doc_ids=doc_ids,
                contents=contents,
                metadatas=metadatas,
            )

        if not doc_ids:
            print("[rag] No valid documents to index")
            return 0

        print(f"[rag] Generating embeddings for {len(doc_ids)} chunks...")
        embeddings = self.llm.embed_batch(contents)

        self.vector_store.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[rag] Indexed {len(doc_ids)} chunks")
        return len(doc_ids)

    def index_all_markdown(self, base_dir: Path) -> int:
        """Index every .md file under base_dir recursively."""
        doc_ids: list[str] = []
        contents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        md_files = sorted(
            p for p in base_dir.rglob("*.md")
            if p.is_file() and "chromadb" not in p.parts and not any(part.startswith(".") for part in p.parts)
        )

        if not md_files:
            print("[rag] No markdown files found to index")
            return 0

        print(f"[rag] Found {len(md_files)} markdown files under {base_dir}")

        for md_file in md_files:
            if "_error" in md_file.stem:
                continue

            content = md_file.read_text(encoding="utf-8")
            if "Duplicate of:" in content or not content.strip():
                continue

            rel_path = md_file.relative_to(base_dir)
            safe_parts = [part.replace(":", "_") for part in rel_path.with_suffix("").parts]
            base_doc_id = "/".join(safe_parts)
            file_path = str(rel_path)
            parent_folder = str(rel_path.parent) if str(rel_path.parent) != "." else "root"

            self._append_chunked_document(
                base_doc_id=base_doc_id,
                content=content,
                metadata={
                    "type": "markdown",
                    "source_folder": parent_folder,
                    "file_path": file_path,
                    "file_name": md_file.name,
                },
                doc_ids=doc_ids,
                contents=contents,
                metadatas=metadatas,
            )

        if not doc_ids:
            print("[rag] No valid markdown files found to index")
            return 0

        print(f"[rag] Generating embeddings for {len(doc_ids)} chunks...")
        embeddings = self.llm.embed_batch(contents)

        self.vector_store.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[rag] Indexed {len(doc_ids)} chunks")
        return len(doc_ids)

    def _rewrite_query_with_context(self, question: str) -> str:
        """Rewrite follow-up question as standalone query using conversation history."""
        if not self.chat_history:
            return question

        history = self._format_recent_history()
        rewrite_prompt = QUERY_REWRITE_PROMPT.format(history=history, question=question)

        rewritten = self.llm.ask(
            "You are a helpful assistant that rewrites questions.",
            rewrite_prompt,
            reasoning=False,
        ).strip()

        print(f"[rag] Rewritten query: {rewritten}")
        return rewritten if rewritten else question

    def retrieve(self, query: str, use_history: bool = True) -> list[dict[str, Any]]:
        # Rewrite query if there's conversation history
        search_query = self._rewrite_query_with_context(query) if use_history else query

        query_embedding = self.llm.embed(search_query)
        return self.vector_store.query(
            query_text=search_query,
            query_embedding=query_embedding,
            n_results=self.n_results,
        )

    def answer(self, question: str, stream: bool = False) -> str:
        retrieved = self.retrieve(question)
        history = self._format_recent_history()

        if not retrieved:
            return self.llm.ask(
                RAG_SYSTEM_PROMPT,
                (
                    "No relevant documents found.\n\n"
                    f"RECENT CONVERSATION:\n{history}\n\n"
                    f"USER QUESTION: {question}\n\n"
                    "Please indicate that you don't have enough information."
                ),
                reasoning=False,
            )

        context_parts: list[str] = []
        total_chars = 0
        for doc in retrieved:
            label = doc["metadata"].get("parent_doc_id", doc["doc_id"])
            file_path = doc["metadata"].get("file_path", "")
            header = f"[Document {label}]"
            if file_path:
                header += f" ({file_path})"
            doc_text = f"{header}\n{doc['content']}"
            if total_chars + len(doc_text) > self.max_context_chars:
                break
            context_parts.append(doc_text)
            total_chars += len(doc_text)

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            history=history,
            question=question,
        )

        return self.llm.ask(
            RAG_SYSTEM_PROMPT,
            user_prompt,
            reasoning=False,
            stream=stream,
            stream_label="rag",
        )

    def chat_loop(self) -> None:
        doc_count = self.vector_store.get_document_count()
        print(f"\n[RAG Chat] {doc_count} documents indexed. Type 'exit' to quit.\n")

        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[RAG Chat] Goodbye!")
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("[RAG Chat] Goodbye!")
                break

            print("\n[RAG Chat] Searching and generating answer...\n")
            answer = self.answer(question, stream=True)
            self.chat_history.append((question, answer))
            # Note: answer already printed during streaming, just add newline
            print()
