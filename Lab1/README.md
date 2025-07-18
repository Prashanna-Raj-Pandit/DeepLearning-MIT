# Lab 1 

**Contents**
**Part 1: Introduction to PyTorch** 
**Part2 : Music Generation using RNN**

![Let's Dance!](http://33.media.tumblr.com/3d223954ad0a77f4e98a7b87136aa395/tumblr_nlct5lFVbF1qhu7oio1_500.gif)

This project implements a character-level music generation model using Recurrent Neural Networks (RNNs), specifically LSTM (Long Short-Term Memory) units, trained on symbolic music data. It is inspired by models used in text generation, adapted to musical sequences ( ABC notation).

### Model architecture

<img src="https://raw.githubusercontent.com/MITDeepLearning/introtodeeplearning/2019/lab1/img/lstm_unrolled-01-01.png" alt="Drawing"/>


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

