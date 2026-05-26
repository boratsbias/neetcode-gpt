"""Train the GPT defined in `model/gpt.py` on a small text corpus.

The model class in `model/gpt.py` is the NeetCode submission, whose `forward`
returns rounded softmax probabilities. For training we need raw logits, so we
subclass it and override `forward` to skip the final softmax/round.

Run:
    python train.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import CharTokenizer
from model.gpt import GPT


class TrainableGPT(GPT):
    """Same architecture as `GPT`, but `forward` returns logits."""

    def forward(self, context: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        token_embeds = self.token_embeddings(context)
        B, T = context.shape
        positions = torch.arange(T, device=context.device)
        pos_embeds = self.pos_embeddings(positions)
        x = token_embeds + pos_embeds
        x = self.blocks(x)
        x = self.final_ln(x)
        return self.vocab_projection(x)


def get_batch(data: torch.Tensor, context_length: int, batch_size: int, device: torch.device):
    # Sample random starting indices, then build (x, y) pairs where y is x shifted by 1.
    ix = torch.randint(0, data.size(0) - context_length - 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, context_length, batch_size, device, iters: int = 20):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, context_length, batch_size, device)
        logits = model(x)
        B, T, V = logits.shape
        losses[k] = F.cross_entropy(logits.view(B * T, V), y.view(B * T)).item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/input.txt")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto|cpu|cuda. MPS is skipped because the NeetCode "
                             "model builds attention masks on CPU.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    print(f"Corpus chars: {len(text):,}  |  Vocab size: {tokenizer.vocab_size}")

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = TrainableGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        model_dim=args.model_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.context_length, args.batch_size, device)
        logits = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, val_data, args.context_length,
                                     args.batch_size, device)
            print(f"step {step:>5d} | train {loss.item():.4f} | val {val_loss:.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": args.context_length,
            "model_dim": args.model_dim,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
        },
        "char_to_int": tokenizer.char_to_int,
        "int_to_char": tokenizer.int_to_char,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved checkpoint -> {args.checkpoint}")


if __name__ == "__main__":
    main()
