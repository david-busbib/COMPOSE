# Checkpoint Manifest

## Files

| File | Size | Description |
|---|---|---|
| `best_model.pt` | ~35 GB | Full COMPOSE model (epoch 40, val_loss=1.1245) |
| `encoder_only.pt` | ~331 MB | Paper-graph encoder only (enc1) |

## Download

```bash
bash scripts/download_checkpoint.sh
```

Or manually:
```bash
huggingface-cli download TheNisso/compose-checkpoint --repo-type model --local-dir checkpoints/
```

## Provenance

Trained with theorem-finetuned (thmft) embeddings via `scripts/train.sh`.
Epoch 40 is the canonical evaluation checkpoint (val_loss=1.1245).
