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
- Cite document IDs when referencing specific information
- If the documents don't contain relevant information, say so clearly
- Be concise but comprehensive
"""

RAG_USER_PROMPT_TEMPLATE = """Based on the following research documents, answer the user's question.

DOCUMENTS:
{context}

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
        max_embed_chars: int = 1800,
        chunk_overlap_chars: int = 200,
    ):
        """
        Initialize RAG system.

        Args:
            llm: LLM client for chat and embeddings
            vector_store: Vector store for document retrieval
            n_results: Number of documents to retrieve per query
            max_context_chars: Maximum context length for LLM
        """
        self.llm = llm
        self.vector_store = vector_store
        self.n_results = n_results
        self.max_context_chars = max_context_chars
        self.max_embed_chars = max_embed_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def _normalize_for_embedding(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r" {3,}", " ", text)
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
                if split_at > start + (self.max_embed_chars // 2):
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
            chunk_metadata.update({
                "parent_doc_id": base_doc_id,
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
            })
            doc_ids.append(chunk_doc_id)
            contents.append(chunk)
            metadatas.append(chunk_metadata)

    def index_documents(
        self,
        summaries_dir: Path,
        index_path: Path | None = None,
    ) -> int:
        """
        Index document summaries into the vector store.

        Args:
            summaries_dir: Directory containing doc_*.md files
            index_path: Optional path to index.json for metadata

        Returns:
            Number of documents indexed
        """
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

            # Skip duplicates
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

        print(f"[rag] Generating embeddings for {len(doc_ids)} documents...")
        embeddings = self.llm.embed_batch(contents)

        self.vector_store.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[rag] Indexed {len(doc_ids)} documents")
        return len(doc_ids)

    def index_all_markdown(self, base_dir: Path) -> int:
        """
        Index all markdown files from multiple research folders.

        Searches for:
        - base_dir/**/summary/doc_*.md
        - base_dir/**/summary/batch_*.md
        - base_dir/**/final.md

        Args:
            base_dir: Root directory containing research_* folders

        Returns:
            Number of documents indexed
        """
        doc_ids: list[str] = []
        contents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        # Collect all md files
        patterns = [
            ("**/summary/doc_*.md", "doc"),
            ("**/summary/batch_*.md", "batch"),
            ("**/final.md", "final"),
        ]

        seen_paths: set[Path] = set()

        for pattern, doc_type in patterns:
            for md_file in base_dir.glob(pattern):
                if md_file in seen_paths:
                    continue
                seen_paths.add(md_file)

                if "_error" in md_file.stem:
                    continue

                content = md_file.read_text(encoding="utf-8")

                # Skip duplicates and empty
                if "Duplicate of:" in content or not content.strip():
                    continue

                # Create unique doc_id from path
                rel_path = md_file.relative_to(base_dir)
                # e.g., research_fe/summary/doc_001.md → research_fe:doc_001
                parent_folder = rel_path.parts[0] if len(rel_path.parts) > 1 else "root"
                doc_id = f"{parent_folder}:{md_file.stem}"

                self._append_chunked_document(
                    base_doc_id=doc_id,
                    content=content,
                    metadata={
                        "type": doc_type,
                        "source_folder": parent_folder,
                        "file_path": str(rel_path),
                    },
                    doc_ids=doc_ids,
                    contents=contents,
                    metadatas=metadatas,
                )

        if not doc_ids:
            print("[rag] No markdown files found to index")
            return 0

        print(f"[rag] Found {len(doc_ids)} markdown files to index")
        print(f"[rag] Generating embeddings...")
        embeddings = self.llm.embed_batch(contents)

        self.vector_store.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"[rag] Indexed {len(doc_ids)} documents")
        return len(doc_ids)

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        query_embedding = self.llm.embed(query)
        return self.vector_store.query(
            query_text=query,
            query_embedding=query_embedding,
            n_results=self.n_results,
        )

    def answer(self, question: str, stream: bool = False) -> str:
        """
        Answer a question using retrieved documents.

        Args:
            question: User's question
            stream: Whether to stream the response

        Returns:
            Generated answer
        """
        retrieved = self.retrieve(question)

        if not retrieved:
            return self.llm.ask(
                RAG_SYSTEM_PROMPT,
                f"No relevant documents found. User question: {question}\n\n"
                "Please indicate that you don't have enough information.",
                reasoning=False,
            )

        context_parts: list[str] = []
        total_chars = 0
        for doc in retrieved:
            doc_text = f"[Document {doc['doc_id']}]\n{doc['content']}"
            if total_chars + len(doc_text) > self.max_context_chars:
                break
            context_parts.append(doc_text)
            total_chars += len(doc_text)

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
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
        """Run interactive chat loop."""
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
            print(f"\n{answer}\n")
