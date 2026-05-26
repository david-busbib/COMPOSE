# Data

Training data is not publicly released due to copyright restrictions on the underlying arXiv and Semantic Scholar content.

**For inference on a single paper, no data is needed** — use `code/run_on_paper.py`.

To retrain or run full evaluation, you need to build the dataset from scratch using your own arXiv + Mathlib data. See:
- `code/build_paper_theorem_embs.py` — embed paper theorem statements
- `code/build_mathlib_graphs.py` — build Mathlib dependency subgraphs
- `code/build_dataset_v7.py` — assemble training samples

Set `COMPOSE_DATA_DIR` to the directory containing your data files. All training and evaluation scripts read from this variable.
