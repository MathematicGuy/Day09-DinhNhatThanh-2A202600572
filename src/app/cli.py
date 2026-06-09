from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.graph import ShoppingAssistant, recommend_improvements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observable shopping assistant CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--recommend", action="store_true")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        output_dir = Path(args.output_dir) if args.output_dir else assistant.settings.traces_dir
        summary = assistant.run_batch(
            Path(args.test_file),
            output_dir,
            rebuild_index=args.rebuild_index,
        )
        print(
            "Batch summary: "
            f"total={summary['total']} "
            f"route_accuracy={summary['route_accuracy']:.2f} "
            f"status_accuracy={summary['status_accuracy']:.2f} "
            f"contains_accuracy={summary['contains_accuracy']:.2f}"
        )
        if args.recommend:
            recommendations = recommend_improvements(summary)
            path = output_dir / "recommendations.json"
            path.write_text(
                json.dumps(recommendations, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            top_priority = recommendations[0]["priority"] if recommendations else "none"
            print(f"Recommendations: count={len(recommendations)} top_priority={top_priority}")
        return

    if args.question:
        trace_file = Path(args.trace_file) if args.trace_file else None
        result = assistant.ask(
            args.question,
            trace_file=trace_file,
            rebuild_index=args.rebuild_index,
        )
        print(result["final_answer"])
        return

    raise SystemExit("Pass --question or --batch.")


if __name__ == "__main__":
    main()
