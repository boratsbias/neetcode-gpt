import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Use torch.multinomial() to choose the next token.
        #    This function simulates a weighted draw from a given list of probabilities
        #    It's similar to picking marbles out of a bag.
        # 2. the given model's output is BEFORE softmax is applied,
        #    and the forward() output has shape batch X time X vocab_size
        # 3. Do not alter the code below, only add to it. This is for maintaining reproducibility in testing.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        res = []

        for i in range(new_chars):
            if len(context.T) > context_length:
                context = context[:, -context_length:]
            prediciton = model(context)
            last_time_step = prediciton[:, -1, :]
            probs = nn.functional.softmax(last_time_step, dim=-1)
            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            next_char = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)
            context = torch.cat((context, next_char), dim=-1)
            res.append(int_to_char[next_char.item()])
        return ''.join(res)

        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox


def _sample_logits(model, new_chars: int, context: torch.Tensor, context_length: int,
                   int_to_char: dict, temperature: float = 1.0) -> str:
    """Stochastic sampler that works with a logits-returning model.

    The NeetCode `Solution.generate` re-seeds its RNG on every step, which makes
    the output deterministic and repetitive. This sampler doesn't re-seed, so it
    produces varied text suitable for actually viewing model output.
    """
    model.eval()
    res = []
    with torch.no_grad():
        for _ in range(new_chars):
            if context.size(1) > context_length:
                context = context[:, -context_length:]
            logits = model(context)[:, -1, :] / max(temperature, 1e-8)
            probs = nn.functional.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1)
            context = torch.cat((context, next_char), dim=-1)
            res.append(int_to_char[next_char.item()])
    return "".join(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--new-chars", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mode", choices=["sample", "neetcode"], default="sample",
                        help="`sample` uses a temperature sampler over logits; "
                             "`neetcode` uses the unmodified Solution.generate.")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found at {args.checkpoint}. Run `python train.py` first.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    int_to_char = ckpt["int_to_char"]
    char_to_int = ckpt["char_to_int"]

    # Encode prompt; fall back to a single newline if any char is OOV.
    try:
        prompt_ids = [char_to_int[c] for c in args.prompt]
    except KeyError:
        prompt_ids = [char_to_int.get("\n", 0)]
    if not prompt_ids:
        prompt_ids = [char_to_int.get("\n", 0)]
    context = torch.tensor([prompt_ids], dtype=torch.long)

    if args.mode == "sample":
        from train import TrainableGPT
        model = TrainableGPT(**cfg)
        model.load_state_dict(ckpt["model_state"])
        output = _sample_logits(model, args.new_chars, context,
                                cfg["context_length"], int_to_char, args.temperature)
    else:
        from model.gpt import GPT
        model = GPT(**cfg)
        model.load_state_dict(ckpt["model_state"])
        output = Solution().generate(model, args.new_chars, context,
                                     cfg["context_length"], int_to_char)

    print(args.prompt + output)


if __name__ == "__main__":
    main()
