"""
extract_search_keywords.py

Pulls the *exact* keywords the agent used while researching literature out of
a Codex CLI rollout JSONL log -- programmatically, not hand-picked and not
truncated. Specifically, for every `tools.web__run(...)` call found in the
log, it structurally parses the call's JSON argument object and collects:

  - every "q" value inside a "search_query" list
      -> what it searched the web for
  - every "pattern" value inside a "find" list
      -> what it grepped for inside a page it had already fetched, i.e. the
         keywords driving the scrape/read step, not just the search step

This is a companion to extract_codex_search_text.py's --actions mode, but
narrower and more precise for this one purpose: --actions mode grabs every
tools.* call (exec_command, apply_patch, web__run, ...) and truncates each
snippet to 280 chars for readability, which cuts off exactly the kind of
multi-query search_query lists this script cares about. This script instead
finds the balanced tools.web__run(...) parens with no length cap, then
json.loads()s the argument object directly and reads the "q"/"pattern"
fields out of the actual structure -- so nothing is retyped by hand and
nothing is cut off mid-query.

Usage:
    python extract_search_keywords.py ~/.codex/sessions/2026/07/29/rollout-....jsonl
    python extract_search_keywords.py path/to/log.jsonl --out search_keywords.md

Then feed the output straight into the heatmap:
    python prompt_vs_keywords.py
"""

import argparse
import json
import re
import sys

_WEB_RUN_RE = re.compile(r"tools\.web__run")


def iter_strings(obj):
    """Recursively yield every string value found inside a JSON-decoded object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v)


def _looks_like_encrypted_blob(s: str) -> bool:
    # Same heuristic as extract_codex_search_text.py: Fernet-encrypted
    # session-state blobs Codex embeds inline, always starting with "gAAAAA"
    # and shaped like one long unbroken run of base64-ish characters.
    if s.startswith("gAAAAA"):
        return True
    if len(s) > 120 and " " not in s:
        non_b64 = re.sub(r"[A-Za-z0-9_\-=]", "", s)
        if len(non_b64) / len(s) < 0.05:
            return True
    return False


def _balanced_paren_arg(s: str, search_from: int):
    """From index `search_from`, find the next '(' and return everything up
    to its matching ')', parens stripped -- balanced on depth, no length
    limit. Returns None if no '(' follows shortly, or the parens never
    balance (e.g. the log line got cut off mid-call).
    """
    i = search_from
    n = len(s)
    while i < n and s[i] not in "(\n":
        i += 1
    if i >= n or s[i] != "(":
        return None
    depth = 0
    j = i
    start = i + 1
    while j < n:
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return s[start:j]
        j += 1
    return None


def extract_web_run_call_args(s: str):
    """Yield the JSON-decoded argument dict of every tools.web__run(...) call
    found in string `s`. Skips calls whose argument isn't valid JSON (e.g. a
    bare variable reference like tools.web__run(previousQuery)).
    """
    for m in _WEB_RUN_RE.finditer(s):
        arg_str = _balanced_paren_arg(s, m.end())
        if arg_str is None:
            continue
        try:
            yield json.loads(arg_str)
        except json.JSONDecodeError:
            continue


def extract_keywords(jsonl_path: str):
    """Scan the whole rollout log and return (search_queries, find_patterns),
    each a list of (line_num, text), in first-seen order, deduplicated.
    """
    search_queries = []
    find_patterns = []
    seen_q, seen_p = set(), set()

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            for s in iter_strings(event):
                s = s.strip()
                if len(s) < 20 or _looks_like_encrypted_blob(s):
                    continue

                for call_args in extract_web_run_call_args(s):
                    if not isinstance(call_args, dict):
                        continue

                    for entry in call_args.get("search_query") or []:
                        q = entry.get("q") if isinstance(entry, dict) else None
                        if q and q not in seen_q:
                            seen_q.add(q)
                            search_queries.append((line_num, q))

                    for entry in call_args.get("find") or []:
                        p = entry.get("pattern") if isinstance(entry, dict) else None
                        if p and p not in seen_p:
                            seen_p.add(p)
                            find_patterns.append((line_num, p))

    return search_queries, find_patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl_path", help="Path to the rollout-*.jsonl session log.")
    parser.add_argument("--out", default="search_keywords.md",
                         help="Output file (default: search_keywords.md). One keyword/pattern per "
                              "block, separated by '---', ready for attention_map.py --raw-blocks.")
    args = parser.parse_args()

    search_queries, find_patterns = extract_keywords(args.jsonl_path)

    if not search_queries and not find_patterns:
        print("No tools.web__run search_query/find calls found in this log. Check the path, "
              "or confirm this session actually used the web tool.")
        sys.exit(1)

    blocks = [q for _, q in search_queries] + [p for _, p in find_patterns]

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\n---\n".join(blocks) + "\n")

    print(f"Found {len(search_queries)} unique search quer{'y' if len(search_queries) == 1 else 'ies'} "
          f"and {len(find_patterns)} unique find pattern(s).")
    print(f"Wrote {len(blocks)} keyword block(s) to {args.out}")
    print("Now run:\n    python prompt_vs_keywords.py")


if __name__ == "__main__":
    main()
