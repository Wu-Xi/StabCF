# StabCF: A Stabilized Training Method for Collaborative Filtering

[![Paper](https://img.shields.io/badge/Paper-KDD%2726-blue)](https://link-to-your-paper.pdf)

This repository provides the **PyTorch implementation** for our paper:

> **StabCF: A Stabilized Training Method for Collaborative Filtering**  
> **Xi Wu**, Wenzhe Zhang, Liangwei Yang, Yi Zhao, Jiquan Peng, Jibing Gong  
> Accepted at *The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026)*

---

## 📌 Introduction

Collaborative Filtering (CF) models trained on implicit feedback are commonly optimized using sampling-based paradigms, most notably Bayesian Personalized Ranking (BPR).  
While such approaches have achieved strong empirical performance, **their training stability has received little attention**.

In practice, we observe that training instability in collaborative filtering mainly arises from two sources:

- **Unreliable positive samples**  
  A single observed interaction may not faithfully reflect a user’s true preference due to noise, sparsity, or accidental behaviors in implicit feedback.

- **Inconsistent negative samples**  
  Negatives sampled from the vast unobserved space often exhibit highly fluctuating hardness, leading to unstable and noisy gradient updates during training.

Together, these issues result in unstable optimization dynamics, slower convergence, and fluctuating recommendation performance.

To address this problem, we propose **StabCF**, a **stabilized training framework** for collaborative filtering.  
Instead of directly optimizing on raw training triplets \((u, i, j)\), StabCF synthesizes **context-aware positive–negative pairs** by jointly enhancing positive reliability and negative consistency.  
This design smooths the training dynamics and leads to more stable and effective model optimization.

<p align="center">
  <img src="assets/Framework1.png" alt="Framework of StabCF" width="700">
</p>

---

## 🚀 Key Features

- **Stability-Oriented Training Framework**  
  StabCF explicitly targets training instability in collaborative filtering by stabilizing both positive and negative learning signals.

- **Context-Aware Sample Synthesis**  
  Raw training triplets are replaced with synthesized \((u, i^\*, j^\*)\) pairs, resulting in smoother optimization dynamics.

- **Model-Agnostic and Plug-and-Play**  
  Easily integrates with popular CF backbones such as LightGCN, NGCF, and ApeGNN without modifying model architectures.

- **Consistent Performance and Stability Gains**  
  Achieves superior accuracy together with significantly more stable convergence across multiple datasets and backbones.

---

## ⚙️ Environment Requirements

The code has been tested with **Python 3.8.0** and **PyTorch 2.0.0**.

Install dependencies with:

```bash
pip install -r requirements.txt
