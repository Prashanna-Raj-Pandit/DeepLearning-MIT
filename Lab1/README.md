# Lab 1 

**Contents**
**Part 1: Introduction to PyTorch** 
**Part2 : Training a Character-Level RNN on Music**

This part of the project dives into building a music-generating neural network from scratch using PyTorch. By the end, the model learns to generate sequences of musical tokens — kind of like how text generators predict the next character — except here it's predicting the next note or event in a MIDI sequence.

This was part of an educational lab where I got hands-on with core deep learning concepts while actually training a generative model. It was messy, eye-opening, and very satisfying.

### Model architecture

Input:         (B, L)             # Batch of sequences (token indices)
Embedding:     (B, L, D)          # Each token becomes a D-dim vector
LSTM:          (B, L, H)          # Hidden representations from context
FC Layer:      (B, L, V)          # Project to vocabulary logits

**Here are the big takeways**

* How to structure an LSTM model for sequential prediction tasks.

* What embedding layers do and why we need them when working with index-based inputs.

* How LSTM's hidden and cell states work and how to initialize and carry them through time steps.

* How to batch variable-length sequences, shift them to form input-output pairs, and reshape tensors for loss calculation.

* How to compute cross-entropy loss when your output is a sequence of predictions over a vocabulary.

* What it means to zero out gradients, backpropagate, and step the optimizer during training.

