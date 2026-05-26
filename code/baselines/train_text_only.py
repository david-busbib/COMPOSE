"""
Text-Only Baseline
==================
Fine-tunes DeepSeek-Math with LoRA directly on (prompt → target_text).
No enc1, no enc2, no cross-attention, no graph.

Prompt format varies by target type:

Variant A — predict main theorem (we_prove, theorem_1x, lean_only, informal_with_lean, ...):
  [INST] You are a mathematician. Given a paper and related work, predict the paper's
  main theorem with some lemma use.

  Paper: {phrase}
  Abstract: {full abstract of target paper}

  Related papers:
  - {neighbor_title}: {neighbor_abstract_2_sentences}
  - ...

  Related Mathlib theorems:
  - {theorem_name}: {statement}
  - ...
  [/INST] {target_text}

Variant B — predict abstract (abstract_full):
  [INST] You are a mathematician. Given a paper and related work, write the paper's abstract.

  Paper: {phrase}

  Related papers:
  - {neighbor_title}: {neighbor_abstract_2_sentences}
  - ...

  Related Mathlib theorems:
  - {theorem_name}: {statement}
  - ...
  [/INST] {target_text}

Usage:
  python3 train_text_only.py --data /path/to/v7.jsonl --checkpoint_dir /path/to/ckpts
"""

import os
import json
import argparse
import logging
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

_REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE          = os.environ.get('COMPOSE_DATA_DIR', os.path.join(_REPO_ROOT, 'data'))
DECODER_MODEL = 'deepseek-ai/deepseek-math-7b-instruct'
MISTRAL_MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ABSTRACT_SOURCES = {'abstract_full'}
THEOREM_SOURCES  = {
    'we_prove', 'we_show', 'we_establish', 'theorem_1x', 'theorem_letter',
    'main_theorem_label', 'we_characterize', 'main_result_of', 'our_main',
    'in_this_paper', 'claim_with_flow', 'lean_only', 'informal_with_lean',
    'lemma_in_context', 'we_prove_no_that', 'purpose', 'no_paper_info'
}

MAX_NEIGHBORS  = 3
MAX_MATHLIB    = 3
MAX_PROMPT_LEN = 1024
MAX_TARGET_LEN = 512


def get_target_node(sample):
    """Find the target paper's node in the subgraph."""
    sg_id = sample['paper_subgraph']['subgraph_id']
    target_id = sg_id.split('_h2')[0] if '_h2' in sg_id else sg_id
    nodes = sample['paper_subgraph'].get('nodes') or []
    for n in nodes:
        if target_id in (n.get('paper_id') or ''):
            return n
    return nodes[0] if nodes else {}


def get_neighbor_nodes(sample):
    """Get non-target paper nodes."""
    sg_id = sample['paper_subgraph']['subgraph_id']
    target_id = sg_id.split('_h2')[0] if '_h2' in sg_id else sg_id
    nodes = sample['paper_subgraph']['nodes']
    return [n for n in nodes if target_id not in n.get('paper_id', '')]


def first_two_sentences(text):
    """Return first 2 sentences of text."""
    if not text:
        return ''
    sents = text.replace('\n', ' ').split('. ')
    return '. '.join(sents[:2]).strip()


def build_prompt(sample):
    """Build the full prompt string for a sample."""
    phrase      = sample.get('phrase', '') or sample.get('target_title', '')
    src         = sample.get('target_text_source', '')

    # Target paper abstract
    target_node = get_target_node(sample) or {}
    abstract_raw = target_node.get('summery_text') or ''
    abstract    = (abstract_raw if isinstance(abstract_raw, str) else ' '.join(abstract_raw)).strip()

    # Neighbor papers (first 2 sentences of abstract)
    neighbors = get_neighbor_nodes(sample)[:MAX_NEIGHBORS]
    neighbor_lines = []
    for n in neighbors:
        title_raw = n.get('title_text', '') or ''
        title = (title_raw if isinstance(title_raw, str) else ' '.join(title_raw)).strip()
        summ_raw = n.get('summery_text', '') or ''
        summ  = first_two_sentences(summ_raw if isinstance(summ_raw, str) else ' '.join(summ_raw))
        if title:
            neighbor_lines.append(f"- {title}: {summ}" if summ else f"- {title}")

    # Mathlib theorems
    mathlib_thms = sample.get('mathlib_subgraph', {}).get('subgraph_theorems', [])
    mathlib_lines = []
    for t in mathlib_thms[:MAX_MATHLIB]:
        name  = t.get('name', '')
        stmt  = str(t.get('statement', '')).strip()[:200]
        if name and stmt:
            mathlib_lines.append(f"- {name}: {stmt}")
        elif name:
            mathlib_lines.append(f"- {name}")

    neighbors_block = '\n'.join(neighbor_lines) if neighbor_lines else '(none)'
    mathlib_block   = '\n'.join(mathlib_lines)  if mathlib_lines  else '(none)'

    if src in ABSTRACT_SOURCES:
        instruction = "write the paper's abstract."
        prompt = (
            f"[INST] You are a mathematician. Given a paper and related work, {instruction}\n\n"
            f"Paper: {phrase}\n\n"
            f"Related papers:\n{neighbors_block}\n\n"
            f"Related Mathlib theorems:\n{mathlib_block}\n"
            f"[/INST]"
        )
    else:
        instruction = "predict the paper's main theorem with some lemma use."
        prompt = (
            f"[INST] You are a mathematician. Given a paper and related work, {instruction}\n\n"
            f"Paper: {phrase}\n"
            f"Abstract: {abstract}\n\n"
            f"Related papers:\n{neighbors_block}\n\n"
            f"Related Mathlib theorems:\n{mathlib_block}\n"
            f"[/INST]"
        )

    return prompt


class TextOnlyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_samples=None):
        self.tokenizer  = tokenizer
        self.samples    = []

        logger.info(f"Loading dataset: {data_path}")
        with open(data_path) as f:
            for line in f:
                s = json.loads(line)
                self.samples.append(s)
                if max_samples and len(self.samples) >= max_samples:
                    break
        logger.info(f"  Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample     = self.samples[idx]
        prompt     = build_prompt(sample)
        target     = sample.get('target_text', '')
        full_text  = prompt + ' ' + target + self.tokenizer.eos_token
        return {'text': full_text, 'prompt': prompt, 'target': target}


def collate_fn(batch, tokenizer, max_len=MAX_PROMPT_LEN + MAX_TARGET_LEN):
    texts = [b['text'] for b in batch]
    enc   = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    # Build labels: -100 for prompt tokens, real ids for target tokens
    prompts     = [b['prompt'] for b in batch]
    prompt_encs = tokenizer(prompts, padding=False, truncation=True, max_length=MAX_PROMPT_LEN)
    prompt_lens = [len(p) for p in prompt_encs['input_ids']]

    labels = enc['input_ids'].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100  # mask prompt tokens

    enc['labels'] = labels
    return enc


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Load tokenizer + model
    decoder_model = args.decoder_model
    logger.info(f"Loading decoder: {decoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        decoder_model,
        torch_dtype=torch.float32,
        device_map='auto',
    )
    model.gradient_checkpointing_enable()

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    dataset = TextOnlyDataset(args.data, tokenizer, max_samples=args.max_train)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # Optimizer — Mistral is already instruction-tuned, use lower LoRA lr
    lora_lr = 1e-7 if 'mistral' in decoder_model.lower() else 1e-6
    other_lr = args.lr
    lora_params  = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    optimizer = AdamW([
        {'params': lora_params,  'lr': lora_lr,  'weight_decay': 0.05},
        {'params': other_params, 'lr': other_lr, 'weight_decay': 1e-2},
    ])

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    loss_log_path = os.path.join(args.checkpoint_dir, 'loss_log.csv')
    loss_rows = []

    logger.info(f"Starting training: {args.epochs} epochs, {len(loader)} steps/epoch")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        loss_rows.append({'epoch': epoch, 'train_loss': avg_loss})

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt_path)
            logger.info(f"  Saved: {ckpt_path}")

        # Write loss log
        with open(loss_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss'])
            writer.writeheader()
            writer.writerows(loss_rows)

    logger.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',           default=os.path.join(BASE, 'dual_training_samples_v7.jsonl'))
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--decoder_model',  default=DECODER_MODEL,
                   help='Decoder LM to fine-tune (default: deepseek-ai/deepseek-math-7b-instruct)')
    p.add_argument('--max_train',      type=int, default=None)
    p.add_argument('--batch_size',     type=int, default=4)
    p.add_argument('--epochs',         type=int, default=30)
    p.add_argument('--lr',             type=float, default=2e-5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
