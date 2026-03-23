import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_client import LLMClient

from autosurvey_tools import (
    duckduckgo_search,
    fetch_webpage,
    is_duplicate_document,
    save_json,
    save_text,
)


SYSTEM_PROMPT = """You are a careful research assistant running on a local model.
Return concise, factual, structured answers.
Do not invent sources or URLs.
When asked for JSON, return valid JSON only.
"""

PLANNER_PROMPT = """Convert the user's research request into a JSON spec.
Return JSON only with this schema:
{
  \"topic\": string,
  \"goal\": string,
  \"search_queries\": [string, ...],
  \"must_cover\": [string, ...],
  \"keywords\": [string, ...]
}
Generate 5-8 search queries. Keep them diverse and web-search friendly.
"""

DOC_SUMMARY_PROMPT = """Summarize the document for later synthesis.
Return JSON only with this schema:
{
  \"title\": string,
  \"source_type\": string,
  \"summary\": string,
  \"key_points\": [string, ...],
  \"reliability_notes\": [string, ...],
  \"keywords\": [string, ...]
}
Keep it concise. Prefer 1-2 sentence summary and 3-5 key points.
"""

BATCH_SUMMARY_PROMPT = """You are given multiple document summaries.
Create a markdown batch note with these sections:
# Batch Summary
## Repeated Findings
## New Findings
## Reliability Notes
## Gaps / Next Search Directions
Be concise and remove redundant statements.
"""

FINAL_PROMPT = """Create the final markdown report.
Required sections:
# Final Research Brief
## User Request
## Executive Summary
## Consolidated Findings
## Repeated / Well-Supported Points
## Conflicts or Uncertainties
## Source Notes
## Remaining Gaps
Rules:
- Deduplicate overlapping content.
- Mention support frequency when relevant.
- Be concrete and concise.
"""


@dataclass
class DocRecord:
    doc_id: str
    title: str
    url: str
    final_url: str
    domain: str
    search_query: str
    text_path: str
    html_path: str
    summary_path: str
    duplicate_of: str | None = None
    duplicate_score: float = 0.0


class AutoSurveyAgent:
    def __init__(
        self,
        output_dir: Path,
        llm: LLMClient,
        batch_size: int,
        max_docs: int,
        max_context: int = 262144,
        plan_reasoning: bool = True,
        summary_reasoning: bool = False,
        final_reasoning: bool = True,
    ):
        self.root = output_dir
        self.corpus_raw_html = self.root / "corpus" / "raw_html"
        self.corpus_raw_text = self.root / "corpus" / "raw_text"
        self.summary_dir = self.root / "summary"
        self.final_path = self.root / "final.md"
        self.index_path = self.summary_dir / "index.json"
        self.plan_path = self.summary_dir / "plan.json"
        self.request_path = self.summary_dir / "request.txt"
        self.llm = llm
        self.batch_size = batch_size
        self.max_docs = max_docs
        self.max_context = max_context
        self.plan_reasoning = plan_reasoning
        self.summary_reasoning = summary_reasoning
        self.final_reasoning = final_reasoning
        self.records: list[DocRecord] = []
        self.text_cache: list[str] = []
        self.batch_counter = 0
        self._prepare_dirs()
        self._load_existing_state()

    def _prepare_dirs(self) -> None:
        self.corpus_raw_html.mkdir(parents=True, exist_ok=True)
        self.corpus_raw_text.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing_state(self) -> None:
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.records = [DocRecord(**r) for r in payload.get("records", [])]
        except Exception:
            self.records = []
            return

        self.text_cache = []
        for record in self.records:
            if record.duplicate_of is not None or not record.text_path:
                continue
            path = Path(record.text_path)
            if path.exists() and path.stat().st_size > 0:
                try:
                    self.text_cache.append(path.read_text(encoding="utf-8"))
                except Exception:
                    pass

        existing_batches = sorted(self.summary_dir.glob("batch_*.md"))
        if existing_batches:
            try:
                self.batch_counter = max(int(p.stem.split("_")[1]) for p in existing_batches)
            except Exception:
                self.batch_counter = 0

    def _doc_id(self, index: int) -> str:
        return f"{index:03d}"

    def _summary_name(self, index: int) -> str:
        return f"doc_{index:03d}.md"

    def _batch_name(self, batch_index: int) -> str:
        return f"batch_{batch_index:03d}.md"

    def _is_zero_byte_file(self, path_str: str) -> bool:
        if not path_str:
            return True
        path = Path(path_str)
        if not path.exists():
            return True
        return path.stat().st_size == 0

    def save_request(self, user_request: str) -> None:
        save_text(self.request_path, user_request.strip() + "\n")

    def load_request(self) -> str:
        if not self.request_path.exists():
            raise FileNotFoundError(f"Missing request file: {self.request_path}")
        return self.request_path.read_text(encoding="utf-8").strip()

    def build_plan(self, user_request: str, *, force: bool = False) -> dict[str, Any]:
        if self.plan_path.exists() and not force:
            return json.loads(self.plan_path.read_text(encoding="utf-8"))
        payload = self.llm.ask_json(PLANNER_PROMPT, user_request, reasoning=self.plan_reasoning)
        if not payload.get("search_queries"):
            payload["search_queries"] = [user_request]
        save_json(self.plan_path, payload)
        return payload

    def load_plan(self) -> dict[str, Any]:
        if not self.plan_path.exists():
            raise FileNotFoundError(f"Missing plan file: {self.plan_path}")
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def collect(self, plan: dict[str, Any]) -> None:
        for query in plan.get("search_queries", []):
            if len(self.records) >= self.max_docs:
                break

            results = duckduckgo_search(query, num_results=5)
            for result in results:
                if len(self.records) >= self.max_docs:
                    break
                if self._already_seen_url(result.url):
                    continue
                self._fetch_one(result.title, result.url, query)

    def _already_seen_url(self, url: str) -> bool:
        return any(r.final_url == url or r.url == url for r in self.records)

    def _fetch_one(self, title_hint: str, url: str, query: str) -> None:
        index = len(self.records)
        doc_id = self._doc_id(index)
        print(f"[fetch] {doc_id}: {url[:80]}...")

        try:
            fetched = fetch_webpage(url)
        except Exception as exc:
            print(f"[fetch] {doc_id}: error - {exc}")
            error_path = self.summary_dir / f"doc_{doc_id}_error.md"
            save_text(error_path, f"# Fetch Error\n\nURL: {url}\n\nError: {exc}\n")
            return

        is_dup, dup_score = is_duplicate_document(fetched.text, self.text_cache)
        if is_dup:
            dup_idx = next(i for i, t in enumerate(self.text_cache) if is_duplicate_document(fetched.text, [t])[0])
            duplicate_of = [r for r in self.records if r.duplicate_of is None][dup_idx].doc_id
            print(f"[fetch] {doc_id}: duplicate of {duplicate_of} (score={dup_score:.3f})")
            record = DocRecord(
                doc_id=doc_id,
                title=fetched.title or title_hint or "Untitled",
                url=url,
                final_url=fetched.final_url,
                domain=fetched.domain,
                search_query=query,
                text_path="",
                html_path="",
                summary_path=str((self.summary_dir / self._summary_name(index)).resolve()),
                duplicate_of=duplicate_of,
                duplicate_score=dup_score,
            )
            duplicate_note = (
                f"# Document {doc_id}\n\n"
                f"- URL: {url}\n"
                f"- Final URL: {fetched.final_url}\n"
                f"- Duplicate of: {duplicate_of}\n"
                f"- Similarity: {dup_score:.3f}\n"
            )
            save_text(self.summary_dir / self._summary_name(index), duplicate_note)
            self.records.append(record)
            self._flush_index()
            return

        html_path = self.corpus_raw_html / f"{doc_id}.html"
        text_path = self.corpus_raw_text / f"{doc_id}.txt"
        save_text(html_path, fetched.html)
        save_text(text_path, fetched.text)

        record = DocRecord(
            doc_id=doc_id,
            title=fetched.title or title_hint or "Untitled",
            url=url,
            final_url=fetched.final_url,
            domain=fetched.domain,
            search_query=query,
            text_path=str(text_path.resolve()),
            html_path=str(html_path.resolve()),
            summary_path=str((self.summary_dir / self._summary_name(index)).resolve()),
        )
        self.records.append(record)
        self.text_cache.append(fetched.text)
        self._flush_index()

    def summarize(self, *, overwrite: bool = False) -> None:
        kept_records = [r for r in self.records if r.duplicate_of is None]
        skipped = [r for r in self.records if r.duplicate_of is not None]
        invalid_empty = [
            r for r in kept_records if self._is_zero_byte_file(r.text_path) or self._is_zero_byte_file(r.html_path)
        ]
        valid_records = [
            r for r in kept_records if not (self._is_zero_byte_file(r.text_path) or self._is_zero_byte_file(r.html_path))
        ]

        if skipped:
            print(f"[info] skipping {len(skipped)} duplicate(s): {[r.doc_id for r in skipped]}")
        if invalid_empty:
            print(f"[info] skipping {len(invalid_empty)} empty document(s): {[r.doc_id for r in invalid_empty]}")

        for record in valid_records:
            summary_path = Path(record.summary_path)
            if summary_path.exists() and summary_path.stat().st_size > 0 and not overwrite:
                print(f"[info] summary exists, skipping {record.doc_id}")
                continue

            print(f"[info] summarizing {record.doc_id}: {record.title[:60]}...")
            text = Path(record.text_path).read_text(encoding="utf-8")
            summary_payload = self.llm.ask_json(
                DOC_SUMMARY_PROMPT,
                json.dumps(
                    {
                        "title_hint": record.title,
                        "url": record.url,
                        "final_url": record.final_url,
                        "domain": record.domain,
                        "title": record.title,
                        "text": text[: self.max_context],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                reasoning=self.summary_reasoning,
                stream=self.llm.stream_summary,
                stream_label=f"summary:{record.doc_id}",
            )
            summary_md = self._render_doc_summary_from_record(record, summary_payload)
            save_text(summary_path, summary_md)

        self.rebuild_batch_summaries(overwrite=overwrite)

    def rebuild_batch_summaries(self, *, overwrite: bool = False) -> None:
        kept_records = [r for r in self.records if r.duplicate_of is None]
        summarized_records = [
            r for r in kept_records
            if Path(r.summary_path).exists() and Path(r.summary_path).stat().st_size > 0
        ]
        if not summarized_records:
            print("[info] no document summaries found; skipping batch summaries")
            return

        self.batch_counter = 0
        for start in range(0, len(summarized_records), self.batch_size):
            batch_records = summarized_records[start : start + self.batch_size]
            batch_number = (start // self.batch_size) + 1
            batch_path = self.summary_dir / self._batch_name(batch_number)
            if batch_path.exists() and batch_path.stat().st_size > 0 and not overwrite:
                print(f"[info] batch summary exists, skipping {batch_path.name}")
                self.batch_counter = batch_number
                continue
            self._write_batch_summary(batch_records, batch_number=batch_number)
            self.batch_counter = batch_number

    def _render_doc_summary_from_record(self, record: DocRecord, payload: dict[str, Any]) -> str:
        lines = [
            f"# Document {record.doc_id}",
            "",
            f"- Title: {payload.get('title') or record.title or 'Untitled'}",
            f"- URL: {record.url}",
            f"- Final URL: {record.final_url}",
            f"- Domain: {record.domain}",
            f"- Search Query: {record.search_query}",
            f"- Source Type: {payload.get('source_type', '')}",
            "",
            "## Summary",
            payload.get("summary", ""),
            "",
            "## Key Points",
        ]
        for point in payload.get("key_points", []):
            lines.append(f"- {point}")
        lines.extend(["", "## Reliability Notes"])
        for note in payload.get("reliability_notes", []):
            lines.append(f"- {note}")
        lines.extend(["", "## Keywords"])
        for keyword in payload.get("keywords", []):
            lines.append(f"- {keyword}")
        return "\n".join(lines).strip() + "\n"

    def _write_batch_summary(self, batch_records: list[DocRecord], *, batch_number: int) -> None:
        summaries = [Path(record.summary_path).read_text(encoding="utf-8") for record in batch_records]
        batch_markdown = self.llm.ask(
            BATCH_SUMMARY_PROMPT,
            "\n\n---\n\n".join(summaries),
            reasoning=False,
            stream=self.llm.stream_summary,
            stream_label=f"batch:{batch_number:03d}",
        )
        save_text(self.summary_dir / self._batch_name(batch_number), batch_markdown)

    def _flush_index(self) -> None:
        save_json(
            self.index_path,
            {
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "records": [asdict(r) for r in self.records],
            },
        )

    def write_final_report(self, user_request: str, plan: dict[str, Any]) -> None:
        kept_records = [r for r in self.records if r.duplicate_of is None]
        batch_summaries = [p.read_text(encoding="utf-8") for p in sorted(self.summary_dir.glob("batch_*.md"))]

        prompt = json.dumps(
            {
                "user_request": user_request,
                "plan": plan,
                "kept_doc_count": len(kept_records),
                "duplicate_count": len([r for r in self.records if r.duplicate_of is not None]),
                "batch_summaries": batch_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        final_markdown = self.llm.ask(FINAL_PROMPT, prompt, reasoning=self.final_reasoning)
        save_text(self.final_path, final_markdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local overnight research agent")
    parser.add_argument("instruction", nargs="?", help="Natural language research instruction")
    parser.add_argument("--output-dir", required=True, help="Root directory for outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=15)
    parser.add_argument("--max-context", type=int, default=16384, help="Max context length for LLM input")
    parser.add_argument(
        "--phase",
        choices=["all", "plan", "collect", "summarize", "final"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--force-plan",
        action="store_true",
        help="Regenerate plan.json even if it already exists",
    )
    parser.add_argument(
        "--overwrite-summaries",
        action="store_true",
        help="Regenerate existing document/batch summaries",
    )
    parser.add_argument(
        "--plan-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /think mode for planning (default: true)",
    )
    parser.add_argument("--stream-summary", action="store_true", help="Stream token output during per-document and batch summaries")
    parser.add_argument(
        "--stream-reasoning",
        action="store_true",
        help="Also print reasoning token chunks when backend exposes them",
    )
    parser.add_argument(
        "--summary-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /think mode for document summaries (default: false)",
    )
    parser.add_argument(
        "--final-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /think mode for final report synthesis (default: true)",
    )
    parser.add_argument(
        "--no-trace-latency",
        action="store_true",
        help="Disable per-call LLM latency logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    llm = LLMClient(
        host=args.host,
        port=args.port,
        stream_summary=args.stream_summary,
        stream_reasoning=args.stream_reasoning,
        trace_latency=not args.no_trace_latency,
    )
    agent = AutoSurveyAgent(
        output_dir=output_dir,
        llm=llm,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        max_context=args.max_context,
        plan_reasoning=args.plan_reasoning,
        summary_reasoning=args.summary_reasoning,
        final_reasoning=args.final_reasoning,
    )

    print(f"[info] output directory = {output_dir}")

    if args.phase == "plan":
        if not args.instruction:
            raise SystemExit("instruction is required for --phase plan")
        agent.save_request(args.instruction)
        print("[info] building plan...")
        agent.build_plan(args.instruction, force=args.force_plan)
        print(f"[done] plan saved to {agent.plan_path}")
        return

    if args.phase == "collect":
        user_request = args.instruction or agent.load_request()
        agent.save_request(user_request)
        plan = agent.build_plan(user_request, force=args.force_plan)
        print("[info] collecting documents...")
        agent.collect(plan)
        print(f"[done] collected {len(agent.records)} record(s)")
        return

    if args.phase == "summarize":
        print("[info] summarizing documents...")
        agent.summarize(overwrite=args.overwrite_summaries)
        print("[done] summaries updated")
        return

    if args.phase == "final":
        user_request = args.instruction or agent.load_request()
        plan = agent.load_plan()
        print("[info] writing final report...")
        agent.write_final_report(user_request, plan)
        print(f"[done] final report saved to {agent.final_path}")
        return

    if not args.instruction:
        raise SystemExit("instruction is required for --phase all")

    agent.save_request(args.instruction)
    print("[info] building plan...")
    plan = agent.build_plan(args.instruction, force=args.force_plan)
    print("[info] collecting documents...")
    agent.collect(plan)
    print("[info] summarizing documents...")
    agent.summarize(overwrite=args.overwrite_summaries)
    print("[info] writing final report...")
    agent.write_final_report(args.instruction, plan)
    print(f"[done] final report saved to {agent.final_path}")


if __name__ == "__main__":
    main()
