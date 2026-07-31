"""Render the explanation for a recorded run, outside the MCP server.

The same report ``explain_run`` returns, available from a shell so a run can be
audited after the agent session that produced it has ended.

    python scripts/mep_report.py                 # latest run, text to stdout
    python scripts/mep_report.py --list          # every recorded run
    python scripts/mep_report.py demo --html     # named run, write report.html
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import mep  # noqa: E402
import mep_rubric  # noqa: E402


def recorded_runs() -> list:
    """Run ids that have a trace on disk, oldest first."""
    if not os.path.isdir(mep.MEP_ROOT):
        return []
    return sorted(
        entry
        for entry in os.listdir(mep.MEP_ROOT)
        if os.path.isfile(os.path.join(mep.MEP_ROOT, entry, "trace.jsonl"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", nargs="?", help="run to explain (default: most recent)")
    parser.add_argument("--list", action="store_true", help="list recorded runs and exit")
    parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="also write the HTML report (default: the run's own report.html)",
    )
    args = parser.parse_args()

    runs = recorded_runs()
    if args.list:
        if not runs:
            print(f"no runs recorded under {mep.MEP_ROOT}")
            return 1
        for run in runs:
            packets = mep_rubric.load(run)
            result = mep_rubric.score(packets)
            violated = sum(f["violated"] for f in result["flags"].values())
            print(
                f"{run:24} {result['n_steps']:>3} steps  "
                f"{result['n_analysis']:>2} analysis  "
                f"{violated}/{len(mep_rubric.RUBRICS)} rubrics violated  "
                f"{'audit-ready' if result['trustworthy'] else 'NOT audit-ready'}"
            )
        return 0

    run_id = args.run_id or (runs[-1] if runs else None)
    if not run_id:
        print(f"no runs recorded under {mep.MEP_ROOT}", file=sys.stderr)
        return 1

    report = None
    if args.html is not None:
        report = args.html or os.path.join(mep.run_directory(run_id), "report.html")

    try:
        print(mep_rubric.explain(run_id, report))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
