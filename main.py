"""Unified CLI entry point for AutoSurvey Agent with RAG Chat."""

import argparse
from pathlib import Path

from llm_client import LLMClient
from vector_store import VectorStore
from rag_system import RAGSystem
from autosurvey_agent import AutoSurveyAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoSurvey Agent with RAG Chat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("instruction", nargs="?", help="Natural language research instruction")
    parser.add_argument("--output-dir", required=True, help="Root directory for outputs")
    parser.add_argument("--host", default="127.0.0.1", help="llama-server host (chat)")
    parser.add_argument("--port", type=int, default=8080, help="llama-server port (chat)")
    parser.add_argument("--embed-host", default=None, help="Embedding server host (if separate)")
    parser.add_argument("--embed-port", type=int, default=None, help="Embedding server port (if separate)")

    # Survey options
    parser.add_argument("--batch-size", type=int, default=5, help="Documents per batch summary")
    parser.add_argument("--max-docs", type=int, default=15, help="Maximum documents to collect")
    parser.add_argument("--max-context", type=int, default=16384, help="Max context length for LLM")

    # Phase control
    parser.add_argument(
        "--phase",
        choices=["all", "plan", "collect", "summarize", "final", "rag"],
        default="all",
        help="Which phase to run (rag = enter RAG chat only)",
    )
    parser.add_argument("--force-plan", action="store_true", help="Regenerate plan.json")
    parser.add_argument("--overwrite-summaries", action="store_true", help="Regenerate summaries")

    # Reasoning options
    parser.add_argument(
        "--plan-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /think mode for planning",
    )
    parser.add_argument(
        "--summary-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use /think mode for document summaries",
    )
    parser.add_argument(
        "--final-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use /think mode for final report",
    )

    # Streaming options
    parser.add_argument("--stream-summary", action="store_true", help="Stream token output")
    parser.add_argument("--stream-reasoning", action="store_true", help="Print reasoning tokens")
    parser.add_argument("--no-trace-latency", action="store_true", help="Disable latency logs")

    # RAG options
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG chat after survey completes",
    )
    parser.add_argument(
        "--rag-results",
        type=int,
        default=5,
        help="Number of documents to retrieve for RAG queries",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents into vector store",
    )

    return parser.parse_args()


def run_survey(args: argparse.Namespace, llm: LLMClient, output_dir: Path) -> None:
    """Run the survey phases."""
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

    # Full pipeline (phase == "all")
    # Skip already completed steps
    final_path = output_dir / "final.md"
    summary_dir = output_dir / "summary"
    has_summaries = summary_dir.exists() and any(summary_dir.glob("doc_*.md"))

    if final_path.exists() and has_summaries:
        print(f"[info] final.md already exists, skipping survey phases")
        return

    agent.save_request(args.instruction)
    print("[info] building plan...")
    plan = agent.build_plan(args.instruction, force=args.force_plan)
    print("[info] collecting documents...")
    agent.collect(plan)
    print("[info] summarizing documents...")
    agent.summarize(overwrite=args.overwrite_summaries)

    if not final_path.exists():
        print("[info] writing final report...")
        agent.write_final_report(args.instruction, plan)
        print(f"[done] final report saved to {agent.final_path}")
    else:
        print(f"[info] final.md already exists, skipping")


def run_rag_chat(args: argparse.Namespace, llm: LLMClient, output_dir: Path) -> None:
    """Initialize and run RAG chat."""
    chromadb_dir = output_dir / "chromadb"
    vector_store = VectorStore(
        persist_dir=chromadb_dir,
        collection_name="research_docs",
        embedding_fn=llm.embed,
    )

    rag = RAGSystem(
        llm=llm,
        vector_store=vector_store,
        n_results=args.rag_results,
    )

    if args.reindex or vector_store.get_document_count() == 0:
        print("[info] Indexing all markdown files under output directory...")
        count = rag.index_all_markdown(output_dir)

        if count == 0:
            print("[error] No documents were indexed. Check your folder structure.")
            return
    else:
        print(f"[info] Using existing index ({vector_store.get_document_count()} documents)")

    rag.chat_loop()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Auto-detect mode: no instruction + existing markdown data → RAG only
    has_any_markdown = any(output_dir.rglob("*.md"))

    if not args.instruction and args.phase == "all":
        if has_any_markdown:
            print("[info] No instruction provided, but documents exist. Entering RAG mode.")
            args.phase = "rag"
        else:
            raise SystemExit("instruction is required (or use --phase rag with existing data)")

    # Initialize LLM client
    llm = LLMClient(
        host=args.host,
        port=args.port,
        embed_host=args.embed_host,
        embed_port=args.embed_port,
        stream_summary=args.stream_summary,
        stream_reasoning=args.stream_reasoning,
        trace_latency=not args.no_trace_latency,
    )

    print(f"[info] output directory = {output_dir}")
    print(f"[info] chat model = {llm.model}")
    if args.embed_host and args.embed_port:
        print(f"[info] embed model = {llm.embed_model} ({args.embed_host}:{args.embed_port})")

    # RAG-only mode
    if args.phase == "rag":
        run_rag_chat(args, llm, output_dir)
        return

    # Run survey phase(s)
    run_survey(args, llm, output_dir)

    # Enter RAG chat after full survey (unless --no-rag)
    if args.phase == "all" and not args.no_rag:
        print("\n" + "=" * 60)
        print("Survey complete! Entering RAG chat mode...")
        print("=" * 60)
        run_rag_chat(args, llm, output_dir)


if __name__ == "__main__":
    main()
