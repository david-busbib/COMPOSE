"""
create the  mathlib grpah 

Build Mathlib theorem dependency graphs from paper citation subgraphs.

For each subgraph in training_samples_with_novelty.jsonl:
1. Find paper nodes that have theorems in the embedding matches file
2. Collect top-K Mathlib matches (score >= 0.84) as seeds
3. For each seed: BFS expand deeply via proof dependencies (graph.json)
4. Output one record per seed in conjecture_statements format + graph_id

Output format mirrors:
  LeanDojo/leandojo_benchmark_4/dataset_with_statements/conjecture_statements_train.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR  = Path(os.environ.get('COMPOSE_DATA_DIR', _REPO_ROOT / 'data'))
LEAN_DIR  = BASE_DIR / "LeanDojo/leandojo_benchmark_4"

NOVELTY_FILE  = BASE_DIR / "training_samples_with_novelty.jsonl"
MATCHES_FILE  = BASE_DIR / "paper_theorems_mathlib_emb_matches_1.jsonl"
GRAPH_FILE    = LEAN_DIR / "processed/graph.json"
THMINFO_FILE  = LEAN_DIR / "processed/theorem_info.json"
PREMINFO_FILE = LEAN_DIR / "processed/premise_info.json"
OUTPUT_FILE   = BASE_DIR / "paper_mathlib_graphs.jsonl"

SCORE_THRESH    = 0.84
TOP_K_MATCHES   = 3   # matches per paper theorem
MAX_PER_HOP     = 8   # max new nodes added per BFS hop
BFS_HOPS        = 6   # hops per graph
SEEDS_PER_GRAPH = 5   # how many seeds per Mathlib subgraph


def load_data():
    print("Loading theorem_to_premises...")
    with open(GRAPH_FILE) as f:
        g = json.load(f)
    theorem_to_premises = g["theorem_to_premises"]

    print("Loading theorem_info...")
    with open(THMINFO_FILE) as f:
        theorem_info = json.load(f)

    print("Loading premise_info...")
    with open(PREMINFO_FILE) as f:
        premise_info = json.load(f)

    print("Loading matches by paper_id...")
    matches_by_paper = defaultdict(list)
    with open(MATCHES_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                matches_by_paper[r["paper_id"]].append(r)
            except Exception:
                pass
    print(f"  {len(matches_by_paper)} papers with matches")

    return theorem_to_premises, theorem_info, premise_info, matches_by_paper


def get_phrase(nodes):
    for node in nodes:
        fields = node.get("fields") or []
        for field_entry in fields:
            if isinstance(field_entry, list):
                for f in field_entry:
                    if f and isinstance(f, str) and not f.startswith("math.") and len(f) > 3:
                        return f
            elif isinstance(field_entry, str) and field_entry:
                return field_entry
    return ""


def thm_entry(name, theorem_info, premise_info):
    info = theorem_info.get(name)
    if info and info.get("state"):
        return {"name": name, "statement": info["state"], "file": info.get("file_path", "")}
    pinfo = premise_info.get(name, {})
    stmt = pinfo.get("code", "")
    file_ = pinfo.get("file_path", info.get("file_path", "") if info else "")
    return {"name": name, "statement": stmt, "file": file_}


def build_graph_for_batch(seed_batch, subgraph_id, batch_idx,
                          paper_nodes, contributing_paper_ids,
                          theorem_to_premises, theorem_info, premise_info):
    """Build one deep graph from a batch of 3-4 seed theorems."""

    # BFS from all seeds simultaneously
    seed_names = [name for name, score in seed_batch]
    all_nodes = set(seed_names)
    frontier = set(seed_names)

    for hop in range(BFS_HOPS):
        next_frontier = set()
        added_this_hop = 0
        for name in frontier:
            for dep in theorem_to_premises.get(name, []):
                if dep not in all_nodes:
                    all_nodes.add(dep)
                    next_frontier.add(dep)
                    added_this_hop += 1
                    if added_this_hop >= MAX_PER_HOP:
                        break
            if added_this_hop >= MAX_PER_HOP:
                break
        frontier = next_frontier
        if not frontier:
            break

    if len(all_nodes) < 3:
        return None

    # seeds first (by score desc), then BFS expansions
    node_list = seed_names + [n for n in all_nodes if n not in set(seed_names)]
    node_idx = {name: i for i, name in enumerate(node_list)}

    # edges: [premise_idx, theorem_idx]
    edge_set = set()
    for name in node_list:
        for dep in theorem_to_premises.get(name, []):
            if dep in node_idx:
                edge_set.add((node_idx[dep], node_idx[name]))
    edges = [list(e) for e in sorted(edge_set)]

    # best seed = highest score in this batch → target_conjecture
    best_seed = seed_batch[0][0]
    target_uses = [
        node_idx[dep]
        for dep in theorem_to_premises.get(best_seed, [])
        if dep in node_idx
    ]

    return {
        "graph_id":          subgraph_id,
        "batch_idx":         batch_idx,
        "paper_ids":         contributing_paper_ids,
        "phrase":            get_phrase(paper_nodes),
        "subgraph_theorems": [thm_entry(n, theorem_info, premise_info) for n in node_list],
        "subgraph_edges":    edges,
        "target_uses":       target_uses,
        "negative_theorems": [],
        "target_conjecture": thm_entry(best_seed, theorem_info, premise_info),
    }


def collect_seeds(subgraph_record, matches_by_paper):
    """Return list of (name, score) seeds and contributing paper_ids."""
    nodes = subgraph_record["subgraph"]["nodes"]
    all_seeds = {}  # name -> best score
    contributing_paper_ids = []

    for node in nodes:
        identifiers = node.get("identifiers") or []
        if not identifiers:
            continue
        paper_id = identifiers[0]
        records = matches_by_paper.get(paper_id, [])
        had_seed = False
        for rec in records:
            for match in rec["matches"][:TOP_K_MATCHES]:
                score = match["score"]
                if score >= SCORE_THRESH:
                    name = match["name"]
                    if name not in all_seeds or score > all_seeds[name]:
                        all_seeds[name] = score
                    had_seed = True
        if had_seed and paper_id not in contributing_paper_ids:
            contributing_paper_ids.append(paper_id)

    # sort seeds by score descending
    seeds = sorted(all_seeds.items(), key=lambda x: -x[1])
    return seeds, contributing_paper_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Process first N subgraphs")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE))
    args = parser.parse_args()

    theorem_to_premises, theorem_info, premise_info, matches_by_paper = load_data()

    skipped = 0
    written = 0
    seen_ids = set()

    with open(NOVELTY_FILE) as fin, open(args.output, "w") as fout:
        for i, line in enumerate(tqdm(fin, desc="Building graphs", total=args.limit)):
            if args.limit and i >= args.limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue

            subgraph_id = rec["subgraph"]["subgraph_id"]
            if subgraph_id in seen_ids:
                continue
            seen_ids.add(subgraph_id)

            seeds, contributing_paper_ids = collect_seeds(rec, matches_by_paper)
            if not seeds:
                skipped += 1
                continue

            paper_nodes = rec["subgraph"]["nodes"]
            graphs_written = 0

            # Split seeds into batches of SEEDS_PER_GRAPH
            batches = [seeds[i:i+SEEDS_PER_GRAPH] for i in range(0, len(seeds), SEEDS_PER_GRAPH)]
            for batch_idx, batch in enumerate(batches):
                result = build_graph_for_batch(
                    batch, subgraph_id, batch_idx,
                    paper_nodes, contributing_paper_ids,
                    theorem_to_premises, theorem_info, premise_info
                )
                if result is None:
                    continue
                fout.write(json.dumps(result) + "\n")
                written += 1
                graphs_written += 1

            if graphs_written == 0:
                skipped += 1

    total_subgraphs = written + skipped  # approximate
    print(f"\nDone.")
    print(f"  Graphs written: {written:,}")
    print(f"  Subgraphs skipped (no seeds): {skipped:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
