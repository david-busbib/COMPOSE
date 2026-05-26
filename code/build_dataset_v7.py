"""
Build v7 dataset: multiple target types per paper.

For each paper with a Mathlib match, generate:
1. informal_only       — just the informal claim (from v6)
2. lean_only           — just the formal Lean lemmas
3. informal_with_lean  — informal claim + formal lemmas combined

Plus keep all original v6 samples (lemma_in_context, claim_with_flow, etc.)

This forces:
- enc1 (paper graph) must be read for informal targets
- enc2 (Mathlib graph) must be read for lean targets
- Both gates must stay open for informal_with_lean
"""

import json
from collections import defaultdict, Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))

# Load Mathlib name -> signature
print("Loading Mathlib signatures...")
mathlib = {}
with open(f"{BASE}/LeanDojo/leandojo_benchmark_4/frenzymath_faiss/mathlib_nl_meta.jsonl") as f:
    for line in f:
        d = json.loads(line)
        mathlib[d['name']] = d['signature'].strip()
print(f"  {len(mathlib)} Mathlib entries")

# Load paper_thm_to_mathlib — group by arxiv_id
print("Loading paper->Mathlib mappings...")
with open(f"{BASE}/paper_thm_to_mathlib.json") as f:
    p2m = json.load(f)

by_arxiv = defaultdict(list)
for stmt_key, mathlib_name in p2m.items():
    arxiv_id = stmt_key.split('_')[0]
    if mathlib_name in mathlib:
        sig = mathlib[mathlib_name]
        if sig:
            by_arxiv[arxiv_id].append(sig)
print(f"  {len(by_arxiv)} arxiv ids with Lean signatures")

# Load v6 — pick best informal claim per paper
print("Loading v6 dataset...")
PRIORITY = [
    'we_prove', 'we_show', 'we_establish', 'we_demonstrate',
    'theorem_1x', 'theorem_letter', 'main_theorem_label',
    'we_characterize', 'main_result_of', 'our_main',
    'in_this_paper', 'abstract_full'
]

best_by_arxiv = {}   # arxiv_id -> best v6 sample
all_v6 = []
with open(f"{BASE}/dual_training_samples_v6.jsonl") as f:
    for line in f:
        d = json.loads(line)
        all_v6.append(d)
        arxiv_id = d.get('target_arxiv_id', '')
        src = d.get('target_text_source', '')
        if arxiv_id not in best_by_arxiv:
            best_by_arxiv[arxiv_id] = d
        else:
            cur_src = best_by_arxiv[arxiv_id].get('target_text_source', '')
            cur_pri = PRIORITY.index(cur_src) if cur_src in PRIORITY else 999
            new_pri = PRIORITY.index(src) if src in PRIORITY else 999
            if new_pri < cur_pri:
                best_by_arxiv[arxiv_id] = d

print(f"  {len(all_v6)} total v6 samples")
print(f"  {len(best_by_arxiv)} unique papers")

# Build v7
print("Building v7 dataset...")
v7 = []

# 1. Keep all original v6 samples
for s in all_v6:
    v7.append(s)

# 2. For papers with Mathlib matches, add new target types
new_lean_only = 0
new_informal_with_lean = 0

for arxiv_id, best_sample in best_by_arxiv.items():
    lean_sigs = by_arxiv.get(arxiv_id, [])
    if not lean_sigs:
        continue

    informal_claim = best_sample.get('target_text', '').strip()
    sigs = lean_sigs[:8]  # cap at 8 lemmas
    lemmas_str = '\n'.join(f'- {sig}' for sig in sigs)

    # Target type 2: lean_only — just the formal lemmas
    lean_sample = dict(best_sample)
    lean_sample['target_text'] = f"Key lemmas:\n{lemmas_str}"
    lean_sample['target_text_source'] = 'lean_only'
    lean_sample['mathlib_signatures'] = sigs
    v7.append(lean_sample)
    new_lean_only += 1

    # Target type 3: informal_with_lean — combined
    if informal_claim:
        combined_sample = dict(best_sample)
        combined_sample['target_text'] = f"{informal_claim}\n\nKey lemmas:\n{lemmas_str}"
        combined_sample['target_text_source'] = 'informal_with_lean'
        combined_sample['mathlib_signatures'] = sigs
        v7.append(combined_sample)
        new_informal_with_lean += 1

print(f"\nv7 dataset stats:")
print(f"  Total samples:            {len(v7)}")
print(f"  Original v6 samples:      {len(all_v6)}")
print(f"  New lean_only:            {new_lean_only}")
print(f"  New informal_with_lean:   {new_informal_with_lean}")

sources = Counter(s['target_text_source'] for s in v7)
print(f"\nTarget source distribution:")
for src, cnt in sources.most_common():
    print(f"  {src}: {cnt} ({cnt/len(v7)*100:.1f}%)")

# Show examples
print("\n--- lean_only example ---")
for s in v7:
    if s['target_text_source'] == 'lean_only':
        print(f"ArXiv: {s['target_arxiv_id']}")
        print(f"Target:\n{s['target_text'][:300]}")
        break

print("\n--- informal_with_lean example ---")
for s in v7:
    if s['target_text_source'] == 'informal_with_lean':
        print(f"ArXiv: {s['target_arxiv_id']}")
        print(f"Target:\n{s['target_text'][:400]}")
        break

# Write
out_path = f"{BASE}/dual_training_samples_v7.jsonl"
print(f"\nWriting to {out_path}...")
with open(out_path, 'w') as f:
    for s in v7:
        f.write(json.dumps(s) + '\n')
print("Done.")
