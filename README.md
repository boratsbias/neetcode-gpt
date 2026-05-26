# My GPT — Built from Scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io)
> Built by **Anubrat Bora** on May 25, 2026

Every file in this project is code I wrote and submitted while completing the NeetCode ML course.
The problems progressively build from gradient descent fundamentals all the way to a working GPT.

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py             Self-attention head
  multi_head_attention.py  Multi-headed attention
  transformer.py           Transformer block
  gpt.py                   GPT model

data/           Data pipeline
  tokenizer.py             Character-level tokenizer
  dataset.py               Batch loader (NeetCode submission)
  nlp_preprocessing.py     NLP preprocessing
  input.txt                Training corpus (Shakespeare excerpt)

train.py        GPT training loop
generate.py     Text generation (also contains the NeetCode Solution class)

foundations/    Neural network primitives built from scratch
  gradient_descent.py, linear_regression.py, linear_regression_training.py,
  pytorch_basics.py, digit_classifier.py, sentiment.py
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py                       # trains on data/input.txt -> checkpoint.pt
python generate.py --new-chars 400    # samples 400 chars from the trained model
```

## Course

This project was built by completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning):
- Math Foundations (gradient descent, activations, loss functions)
- Neural Networks from scratch (neuron, backprop, MLP)
- PyTorch fundamentals
- NLP pipeline (embeddings, tokenization, attention)
- Transformer architecture
- GPT model + text generation
