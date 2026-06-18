from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import analyze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kpop-scope", description="K-pop oriented MIR analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Analyze a local audio file")
    p.add_argument("audio", type=str, help="Path to local audio file")
    p.add_argument("--output", "-o", type=str, default="outputs/kpop_scope_analysis", help="Output directory")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    p.add_argument("--stems", action="store_true", help="Run Demucs stem separation if available")
    p.add_argument("--plots", action="store_true", help="Generate figures (enabled by default; kept for README compatibility)")
    p.add_argument("--no-plots", action="store_true", help="Disable figure generation")
    p.add_argument("--max-duration", type=float, default=None, help="Analyze first N seconds only")
    p.add_argument("--mert", action="store_true", help="Extract MERT embedding during tagging")
    p.add_argument("--classifier", type=str, default=None, help="Path to trained K-pop classifier checkpoint")
    p.add_argument("--no-tag-fallback", action="store_true", help="Fail if MERT/classifier tagging is unavailable instead of using acoustic prior")
    p.add_argument("--report-mode", choices=["tag_only", "evidence_grounded"], default=None, help="Report generation mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        result = analyze(
            args.audio,
            output_dir=args.output,
            config_path=args.config,
            use_stems=args.stems,
            make_plots=not args.no_plots,
            max_duration=args.max_duration,
            use_mert=args.mert,
            classifier_path=args.classifier,
            fallback_to_acoustic_prior=not args.no_tag_fallback,
            report_mode=args.report_mode,
        )
        out = Path(args.output)
        print(f"Analysis complete: {out.resolve()}")
        print(f"Report: {(out / 'report.md').resolve()}")
        print("Top tags:", ", ".join(result.get("explanations", {}).get("top_tags", [])[:8]))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
